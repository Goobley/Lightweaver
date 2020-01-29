from CmoArray cimport F64View, F64View2D, F64View3D, F64View4D, f64
cdef F64View f64_view(f64[::1] memview) except +
cdef F64View2D f64_view_2(f64[:, ::1] memview) except +
cdef F64View3D f64_view_3(f64[:, :, ::1] memview) except +
cdef F64View4D f64_view_4(f64[:, :, :, ::1] memview) except +