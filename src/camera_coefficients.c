#include <stddef.h>
#include "camera_decode.h"
typedef struct {
    camera_block* plane[3];
    int32_t* values;
    unsigned used;
    camera_tables* tables;
    camera_plan* plan;
    size_t capacity, file_size, blocks;
    camera_allocate allocate;
    void* allocator;
    unsigned next_group;
} camera_output;
static inline void store_camera_value(int32_t* values, unsigned* written, uint64_t* mask,
                                      unsigned k, int32_t coefficient) {
    if (coefficient) {
        *mask |= UINT64_C(1) << k;
        values[(*written)++] = coefficient;
    }
}

#define CUJPEGXL_CAPTURE capture_blocks
#include "jxldec_camera.inc"

// The CUDA group reader handles a single ANS preset. Other entropy models
// use the sparse host reader and the same CUDA reconstruction kernels.
static int prepare_camera_payload(jxl_ctx* ctx, const jxl_vardct_state* v, uint32_t groups,
                                  int single) {
    camera_output* camera = ctx->user;
    const jxl_dec* dist = &v->passes[0].dist;
    size_t histogram = sizeof(camera_group) * groups;
    int supported = !single && !dist->use_prefix && !dist->lz77_enabled && v->num_hf_presets == 1;
    if (supported) {
        unsigned log_bucket = dist->ans[0].log_bucket_size;
        for (unsigned c = 0; c < dist->num_clusters; ++c) {
            jxl_int_config u = dist->configs[c];
            if (dist->ans[c].log_bucket_size != log_bucket || u.split_exponent != 4 ||
                u.msb_in_token != 2 || u.lsb_in_token != 0)
                supported = 0;
        }
        size_t contexts =
            histogram + sizeof(camera_bucket) * dist->num_clusters * (4096u >> log_bucket);
        size_t bytes = (contexts + dist->num_dist + 3) & ~(size_t)3;
        // Four independent warps also need three 32-entry predictor rows each.
        if (supported && bytes - histogram <= 48 * 1024 - 4 * 3 * 32 * sizeof(int) &&
            camera->file_size + 12ul * groups + bytes <= INT32_MAX) {
            *camera->plan = (camera_plan){groups,     v->block_ctx.num_block_clusters,
                                          contexts,   histogram,
                                          log_bucket, bytes - histogram};
            camera->capacity = bytes + camera->file_size + 12ul * groups;
            camera->values = camera->allocate(camera->allocator, camera->capacity);
            if (!camera->values)
                return -1;
            uint8_t* data = (uint8_t*)camera->values;
            camera_bucket* maps = (camera_bucket*)(data + histogram);
            unsigned count = 4096u >> log_bucket;
            _Static_assert(sizeof(camera_bucket) == sizeof(jxl_ans_bucket), "ANS bucket layout");
            for (unsigned c = 0; c < dist->num_clusters; ++c)
                memcpy(maps + c * count, dist->ans[c].buckets, count * sizeof(camera_bucket));
            memcpy(data + contexts, dist->clusters, dist->num_dist);
            memset(data + contexts + dist->num_dist, 0, bytes - contexts - dist->num_dist);
            camera->used = bytes / 4;
            return 0;
        }
    }
    camera->used = 2 * camera->blocks;  // Two scan-order mask words per block.
    camera->capacity = camera->blocks * 65 * sizeof(int32_t);
    camera->values = camera->allocate(camera->allocator, camera->capacity);
    return camera->values ? 0 : -1;
}

static int capture_blocks(jxl_ctx* ctx, jxl_vardct_state* v, const jxl_image_metadata* meta,
                          const jxl_frame_header* fh) {
    camera_output* output = ctx->user;
    camera_tables* tables = output->tables;
    tables->bias_numerator = meta->quant_bias_numerator;
    for (int c = 0; c < 3; ++c) {
        float scale =
            c == 1 ? 1 : powf(0.8f, (float)(c == 0 ? fh->x_qm_scale : fh->b_qm_scale) - 2);
        tables->bias[c] = meta->quant_bias[c];
        memcpy(tables->weights[c], v->dm.matrix_tr[jxl_tr_matrix_index(JXL_TR_DCT8)][c],
               64 * sizeof(float));
        uint16_t const* order = v->passes[0].order[0][c];
        if (!order)
            order = jxl_natural_order(ctx, &v->no, 0);
        if (!order)
            return -1;
        for (unsigned k = 0; k < 64; ++k)
            tables->frequency[c][k] = order[2 * k] * 8 + order[2 * k + 1];
        int cw = v->bw >> v->hs[c];
        for (unsigned y = 0; y < v->bh; ++y) {
            for (int x = 0; x < cw; ++x) {
                camera_block* block = output->plane[c] + y * cw + x;
                const jxl_block_info* bi = &v->block_info[y * v->bw + (x << v->hs[c])];
                block->dc = v->lf[c][y * v->bw + x];
                block->scale =
                    65536.0f / ((float)v->quantizer.global_scale * (float)bi->hf_mul) * scale;
            }
        }
    }
    return 0;
}

int camera_coefficients(const unsigned char* data, size_t len, unsigned width, unsigned height,
                        camera_block* blocks, camera_tables* tables, camera_plan* plan,
                        camera_allocate allocate, void* allocator) {
    unsigned chroma_blocks = ((width + 15) / 16) * ((height + 7) / 8);
    camera_output output = {{blocks, blocks + chroma_blocks, blocks + chroma_blocks * 3},
                            NULL,
                            0,
                            tables,
                            plan,
                            0,
                            len,
                            4ul * chroma_blocks,
                            allocate,
                            allocator,
                            0};
    plan->groups = 0;
    jxl_ctx* ctx = jxl_ctx_new(NULL, NULL, NULL, &output);
    if (!ctx)
        return -1;
    jxl_doc* doc = jxl_doc_open(ctx, data, len);
    jxl_frame_header fh = {0};
    jxl_toc toc = {0};
    jxl_frame_state state = {0};
    jxl_fimage image = {0};
    jxl_br br;
    int status = -1;
    if (!doc || doc->size.width != width || doc->size.height != height ||
        doc->meta.orientation != 1 || doc->meta.have_animation || doc->meta.num_extra ||
        doc->meta.bit_depth.bits_per_sample != 8 || doc->meta.xyb_encoded ||
        doc->meta.colour.white_point != JXL_WP_D65 ||
        doc->meta.colour.primaries != JXL_PRIMARIES_SRGB || doc->meta.colour.want_icc ||
        doc->meta.colour.tf != JXL_TF_SRGB || doc->meta.colour.colour_space != JXLDEC_CS_RGB)
        goto done;
    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);
    if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) || fh.passes.num_passes != 1 ||
        fh.jpeg_upsampling[0] != 0 || fh.jpeg_upsampling[1] != 2 || fh.jpeg_upsampling[2] != 0 ||
        jxl_frame_blocks_w(&fh) != 2 * ((width + 15) / 16) ||
        jxl_frame_blocks_h(&fh) != (height + 7) / 8 || fh.encoding != JXL_ENC_VARDCT ||
        !fh.do_ycbcr || !fh.is_last || fh.frame_type != JXL_FRAME_REGULAR || fh.have_crop ||
        fh.upsampling != 1 || fh.flags & ~JXL_FF_SKIP_ADAPTIVE_LF_SMOOTH || fh.gab.enabled ||
        fh.epf.enabled || jxl_read_toc(ctx, &br, &fh, &toc) ||
        toc.end_off > doc->container.cs_len || toc.total_size > doc->container.cs_len - toc.end_off)
        goto done;
    status = jxl_frame_decode(ctx, doc, &fh, &toc, &state, 0, &image);
done:
    jxl_fimage_free(ctx, &image);
    jxl_frame_state_free(ctx, &state);
    jxl_toc_free(ctx, &toc);
    jxl_frame_header_free(ctx, &fh);
    if (doc)
        jxl_doc_close(doc);
    jxl_ctx_free(ctx);
    return status == 0 ? (int)output.used : -1;
}
