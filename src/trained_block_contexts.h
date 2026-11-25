// Trained on native JPEG XL coefficient tokens from eight RAW development views.
// Subsampled chroma uses the same context in every quant-field bin.
// libjxl and jxl-rs sample that field at different horizontal positions.
#define CUJPEGXL_BLOCK_CONTEXTS 0,0,8,3,3,4,5,5,6,0,8,3,4,4,5,5,11,4,8,4,1,1,1,5,6,0,8,4,9,9,9,9,14,13,14,5,5,5,5,5,11,0,14,5,5,14,14,14,14,7,13,13,13,5,5,5,7,7,13,13,13,5,14,14,2,12,5,12,2,2,2,2,5,5,5,12,2,10,10,10,2,12,5,12,2,2,2,2,5,5,5,12,2,10,10,10,2,12,5,12,2,2,2,2,5,5,5,12,2,10,10,10,2,12,5,12,2,2,2,2,5,5,5,12,2,10,10,10,12,12,5,12,12,2,2,12,5,5,5,12,2,10,10,10,12,12,5,12,12,2,2,12,5,5,5,12,2,10,10,10,12,12,5,12,12,2,2,12,5,5,5,12,2,10,10,10,12,12,5,12,12,2,2,12,5,5,5,12,2,10,10,10
