#define CUJPEGXL_CAPTURE capture_blocks
#include "../third_party/jxldec/jxl.c"

// Three packed planes of consecutive 8x8 blocks, in Cb, Y, Cr order.
typedef struct {
    float *blocks;
    uint32_t width, height;
} camera_output;

static int capture_blocks(jxl_ctx *ctx, jxl_vardct_state *v,
                          const jxl_image_metadata *meta, const jxl_frame_header *fh) {
    camera_output *output = ctx->user;
    float *dest = output->blocks;
    if (v->bw != 2 * ((output->width + 15) / 16) || v->bh != (output->height + 7) / 8)
        return -1;
    const float scales[] = {powf(0.8f, (float)fh->x_qm_scale - 2), 1,
                            powf(0.8f, (float)fh->b_qm_scale - 2)};
    for (int c = 0; c < 3; ++c) {
        if (v->hs[c] != (c != 1) || v->vs[c] != 0) return -1;
        for (uint32_t y = 0; y < v->bh; ++y) {
            for (uint32_t x = 0; x < (v->bw >> v->hs[c]); ++x, dest += 64) {
                jxl_block_info *bi = &v->block_info[y*v->bw + (x << v->hs[c])];
                if (bi->dct_select != JXL_TR_DCT8) return -1;
                float *block = v->coeff[c] + y*8*v->ps + x*8;
                if (bi->hf_nonzero_mask & (1u << c)) {
                    jxl_dequant_varblock(block, v->ps, JXL_TR_DCT8, bi->hf_mul, c,
                        &v->dm, &v->quantizer, scales[c], meta->quant_bias[c], meta->quant_bias_numerator);
                    for (int row = 0; row < 8; ++row)
                        memcpy(dest + row*8, block + row*v->ps, 8*sizeof(float));
                } else memset(dest, 0, 64*sizeof(float));
                dest[0] = v->lf[c][y*v->bw + x];
            }
        }
    }
    return 0;
}

int camera_coefficients(const unsigned char *data, size_t len, unsigned width,
                        unsigned height, float *blocks) {
    camera_output output = {blocks, width, height};
    jxl_ctx *ctx = jxl_ctx_new(NULL, NULL, NULL, &output);
    if (!ctx) return -1;
    jxl_doc *doc = jxl_doc_open(ctx, data, len);
    jxl_frame_header fh = {0};
    jxl_toc toc = {0};
    jxl_frame_state state = {0};
    jxl_fimage image = {0};
    jxl_br br;
    int status = -1;
    if (!doc || doc->size.width != width || doc->size.height != height ||
        doc->meta.orientation != 1 || doc->meta.have_animation || doc->meta.num_extra ||
        doc->meta.bit_depth.bits_per_sample != 8 || doc->meta.xyb_encoded ||
        doc->meta.colour.want_icc || doc->meta.colour.tf != JXL_TF_SRGB ||
        doc->meta.colour.colour_space != JXLDEC_CS_RGB) goto done;
    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);
    if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) ||
        fh.encoding != JXL_ENC_VARDCT || !fh.do_ycbcr || !fh.is_last ||
        fh.frame_type != JXL_FRAME_REGULAR || fh.have_crop || fh.upsampling != 1 ||
        fh.flags & ~JXL_FF_SKIP_ADAPTIVE_LF_SMOOTH || fh.gab.enabled || fh.epf.enabled ||
        jxl_read_toc(ctx, &br, &fh, &toc) || toc.end_off > doc->container.cs_len ||
        toc.total_size > doc->container.cs_len - toc.end_off) goto done;
    status = jxl_frame_decode(ctx, doc, &fh, &toc, &state, 0, &image);
done:
    jxl_fimage_free(ctx, &image);
    jxl_frame_state_free(ctx, &state);
    jxl_toc_free(ctx, &toc);
    jxl_frame_header_free(ctx, &fh);
    if (doc) jxl_doc_close(doc);
    jxl_ctx_free(ctx);
    return status;
}
