The Torch decoder uses the VarDCT coefficient reader from [jxldec](https://github.com/kjk/jxldec).
CMake FetchContent downloads the pinned source by commit and verifies its SHA-256.
The wheel build requires CMake and GNU patch. The wheel contains the compiled reader and its license.

The public jxldec API returns pixels. The GPU reconstruction needs coefficient blocks and entropy sections before that step. The local patch exposes those points to the camera adapter. CMake applies it to the upstream amalgamation in the build directory. To update the dependency, change the commit and digest, rebase the patch, and run the Torch decoder tests.

For a disconnected build, configure the CMake build directory with
`-DFETCHCONTENT_SOURCE_DIR_JXLDEC=/absolute/path/to/pinned/jxldec` before building the wheel.
