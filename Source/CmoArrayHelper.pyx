from CmoArray cimport F64View, F64View2D, F64View3D, F64View4D
from CmoArrayHelper cimport f64

cdef F64View f64_view(f64[::1] memview) except +:
    if memview.shape[0] == 0:
        return F64View()
    return F64View(&memview[0], memview.shape[0])

cdef F64View2D f64_view_2(f64[:, ::1] memview) except +:
    return F64View2D(&memview[0,0], memview.shape[0], memview.shape[1])

cdef F64View3D f64_view_3(f64[:, :, ::1] memview) except +:
    return F64View3D(&memview[0,0,0], memview.shape[0], memview.shape[1], memview.shape[2])

cdef F64View4D f64_view_4(f64[:, :, :, ::1] memview) except +:
    return F64View4D(&memview[0,0,0,0], memview.shape[0], memview.shape[1], memview.shape[2], memview.shape[3])