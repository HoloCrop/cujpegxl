#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Orthonormal 8-point inverse DCT basis, scaled for JPEG XL's mean-valued DC.
__constant__ float basis[8][8] = {
    {1.0000000000f, 1.3870398453f, 1.3065629649f, 1.1758756024f, 1.0000000000f, 0.7856949584f, 0.5411961001f, 0.2758993793f},
    {1.0000000000f, 1.1758756024f, 0.5411961001f, -0.2758993793f, -1.0000000000f, -1.3870398453f, -1.3065629649f, -0.7856949584f},
    {1.0000000000f, 0.7856949584f, -0.5411961001f, -1.3870398453f, -1.0000000000f, 0.2758993793f, 1.3065629649f, 1.1758756024f},
    {1.0000000000f, 0.2758993793f, -1.3065629649f, -0.7856949584f, 1.0000000000f, 1.1758756024f, -0.5411961001f, -1.3870398453f},
    {1.0000000000f, -0.2758993793f, -1.3065629649f, 0.7856949584f, 1.0000000000f, -1.1758756024f, -0.5411961001f, 1.3870398453f},
    {1.0000000000f, -0.7856949584f, -0.5411961001f, 1.3870398453f, -1.0000000000f, -0.2758993793f, 1.3065629649f, -1.1758756024f},
    {1.0000000000f, -1.1758756024f, 0.5411961001f, 0.2758993793f, -1.0000000000f, 1.3870398453f, -1.3065629649f, 0.7856949584f},
    {1.0000000000f, -1.3870398453f, 1.3065629649f, -1.1758756024f, 1.0000000000f, -0.7856949584f, 0.5411961001f, -0.2758993793f}
};

__global__ void inverse_blocks(float *blocks) {
    __shared__ float scratch[64];
    int i = threadIdx.x, x = i % 8, y = i / 8;
    float *block = blocks + (size_t(blockIdx.y)*gridDim.x + blockIdx.x)*64;
    scratch[i] = block[i];
    __syncthreads();
    float row = 0;
    for (int k=0; k<8; ++k) row += basis[x][k]*scratch[y*8+k];
    __syncthreads();
    scratch[i] = row;
    __syncthreads();
    float value = 0;
    for (int k=0; k<8; ++k) value += basis[y][k]*scratch[k*8+x];
    block[i] = value;
}

__device__ float sample(const float *plane, int bw, int x, int y) {
    return plane[((y/8)*bw+x/8)*64 + (y%8)*8 + x%8];
}

__device__ int channel(float value) { return __float2int_rn(fminf(255, fmaxf(0, value*255))); }

__device__ int3 rgb(const float *blocks, int w, int h, int x, int y) {
    if (x<0 || y<0 || x>=w || y>=h) return make_int3(0,0,0);
    int cw = (w+15)/16, plane = cw*((h+7)/8)*64;
    int cx=x/2, neighbor=min((w-1)/2,max(0,cx+((x&1)?1:-1)));
    float cb=.75f*sample(blocks,cw,cx,y)+.25f*sample(blocks,cw,neighbor,y);
    float luma=sample(blocks+plane,2*cw,x,y)+128.f/255;
    float cr=.75f*sample(blocks+3*plane,cw,cx,y)+.25f*sample(blocks+3*plane,cw,neighbor,y);
    return make_int3(channel(luma+1.402f*cr), channel(luma-.344136f*cb-.714136f*cr), channel(luma+1.772f*cb));
}

__device__ int interpolate(int a,int b,int c,int d,int fx,int fy) {
    return ((a*(32-fx)+b*fx)*(32-fy)+(c*(32-fx)+d*fx)*fy+512)/1024;
}

__global__ void render(const float *blocks, const float2 *maps, size_t map_stride,
                       unsigned char *out, int w,int h,int ow,int oh,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=ow*oh) return;
    blocks += size_t(blockIdx.y)*count*64;
    int3 pixel;
    if (maps) {
        float2 coord=maps[blockIdx.y*map_stride+i];
        int u=__float2int_rn(coord.x*32), v=__float2int_rn(coord.y*32);
        int x=u>>5,y=v>>5,fx=u&31,fy=v&31;
        int3 a=rgb(blocks,w,h,x,y),b=rgb(blocks,w,h,x+1,y);
        int3 c=rgb(blocks,w,h,x,y+1),d=rgb(blocks,w,h,x+1,y+1);
        pixel=make_int3(interpolate(a.x,b.x,c.x,d.x,fx,fy),interpolate(a.y,b.y,c.y,d.y,fx,fy),
                        interpolate(a.z,b.z,c.z,d.z,fx,fy));
    } else pixel=rgb(blocks,w,h,i%ow,i/ow);
    size_t index=(size_t(blockIdx.y)*ow*oh+i)*3;
    out[index]=pixel.x; out[index+1]=pixel.y; out[index+2]=pixel.z;
}

void render_camera(float *blocks, const float *maps, size_t map_stride, unsigned char *out,
                   int batch,int count,int w,int h,int ow,int oh,cudaStream_t stream) {
    inverse_blocks<<<dim3(count,batch),64,0,stream>>>(blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    render<<<dim3((ow*oh+255)/256,batch),256,0,stream>>>(blocks,reinterpret_cast<const float2*>(maps),
        map_stride,out,w,h,ow,oh,count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
