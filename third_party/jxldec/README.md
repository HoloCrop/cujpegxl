Source: https://github.com/kjk/jxldec at b062dd5b34a76f29a858b2772a003b4bcd6c8b2a.

The camera integration intercepts VarDCT allocation, group reading and final
capture. The reader exports the compressed AC sections, entropy tables and block
contexts for CUDA. The sparse host path records coefficients in scan order.
The unmodified decoder paths are compiled when CUJPEGXL_CAPTURE is undefined.
