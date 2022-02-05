import numpy as np
cimport numpy as np

cdef extern from "CmoArray.hpp" namespace "Jasnah":
    cdef cppclass Array1NonOwn[T]:
        T* data
        Array1NonOwn()
        Array1NonOwn(T*, size_t)
        Array1NonOwn(Array1Own[T]&)
        Array1NonOwn& operator=(Array1Own[T]&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array2NonOwn[T] reshape(int, int)
        Array3NonOwn[T] reshape(int, int, int)
        Array4NonOwn[T] reshape(int, int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array1Own[T]:
        Array1Own(size_t)
        Array1Own(T, size_t)
        Array1Own(const Array1NonOwn&)
        Array1Own& operator=(const Array1NonOwn&)
        T* data()
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array2NonOwn[T] reshape(int, int)
        Array3NonOwn[T] reshape(int, int, int)
        Array4NonOwn[T] reshape(int, int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array2NonOwn[T]:
        T* data
        Array2NonOwn()
        Array2NonOwn(T*, size_t, size_t)
        Array2NonOwn(Array2Own[T]&)
        Array2NonOwn& operator=(Array2Own[T]&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t) const
        T& operator()(size_t, size_t)
        Array1NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array3NonOwn[T] reshape(int, int, int)
        Array4NonOwn[T] reshape(int, int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array2Own[T]:
        Array2Own(size_t, size_t)
        Array2Own(T, size_t, size_t)
        Array2Own(const Array2NonOwn&)
        Array2Own& operator=(const Array2NonOwn&)
        T* data()
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array3NonOwn[T] reshape(int, int, int)
        Array4NonOwn[T] reshape(int, int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array3NonOwn[T]:
        T* data
        Array3NonOwn()
        Array3NonOwn(T*, size_t, size_t, size_t)
        Array3NonOwn(Array3Own[T]&)
        Array3NonOwn& operator=(Array3Own[T]&) # This function doesn't actually exist, but cython seems to need to believe it does to assign an Arr to a View
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t, size_t) const
        T& operator()(size_t, size_t, size_t)
        Array1NonOwn[T] operator()(size_t, size_t)
        Array2NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array2NonOwn[T] reshape(int, int)
        Array4NonOwn[T] reshape(int, int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array3Own[T]:
        Array3Own(size_t, size_t, size_t)
        Array3Own(T, size_t, size_t, size_t)
        Array3Own(const Array3NonOwn&)
        Array3Own& operator=(const Array3NonOwn&)
        T* data()
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t, size_t)
        Array2NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array2NonOwn[T] reshape(int, int)
        Array4NonOwn[T] reshape(int, int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array4NonOwn[T]:
        T* data
        Array4NonOwn()
        Array4NonOwn(T*, size_t, size_t, size_t, size_t)
        Array4NonOwn(Array4Own[T]&)
        Array4NonOwn& operator=(Array4Own[T]&) # This function doesn't actually exist, but cython seems to need to believe it does to assign an Arr to a View
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t, size_t, size_t) const
        T& operator()(size_t, size_t, size_t, size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t)
        Array3NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array2NonOwn[T] reshape(int, int)
        Array3NonOwn[T] reshape(int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array4Own[T]:
        Array4Own(size_t, size_t, size_t, size_t)
        Array4Own(T, size_t, size_t, size_t, size_t)
        Array4Own(const Array4NonOwn&)
        Array4Own& operator=(const Array4NonOwn&)
        T* data()
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t)
        Array3NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array2NonOwn[T] reshape(int, int)
        Array3NonOwn[T] reshape(int, int, int)
        Array5NonOwn[T] reshape(int, int, int, int, int)

    cdef cppclass Array5NonOwn[T]:
        T* data
        Array5NonOwn()
        Array5NonOwn(T*, size_t, size_t, size_t, size_t, size_t)
        Array5NonOwn(Array5Own[T]&)
        Array5NonOwn& operator=(Array5Own[T]&) # This function doesn't actually exist, but cython seems to need to believe it does to assign an Arr to a View
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t, size_t, size_t, size_t) const
        T& operator()(size_t, size_t, size_t, size_t, size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t, size_t)
        Array3NonOwn[T] operator()(size_t, size_t)
        Array4NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array2NonOwn[T] reshape(int, int)
        Array3NonOwn[T] reshape(int, int, int)
        Array4NonOwn[T] reshape(int, int, int, int)

    cdef cppclass Array5Own[T]:
        Array5Own(size_t, size_t, size_t, size_t, size_t)
        Array5Own(T*, size_t, size_t, size_t, size_t, size_t)
        Array5Own(const Array5NonOwn&)
        Array5Own& operator=(const Array5NonOwn&)
        T* data()
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t, size_t)
        Array3NonOwn[T] operator()(size_t, size_t)
        Array4NonOwn[T] operator()(size_t)
        void fill(T)
        size_t* shape()
        size_t shape(size_t)
        Array1NonOwn[T] flatten()
        Array1NonOwn[T] reshape(int)
        Array2NonOwn[T] reshape(int, int)
        Array3NonOwn[T] reshape(int, int, int)
        Array4NonOwn[T] reshape(int, int, int, int)

ctypedef Array1NonOwn[double] F64View
ctypedef Array1NonOwn[double] F64View1D
ctypedef Array2NonOwn[double] F64View2D
ctypedef Array3NonOwn[double] F64View3D
ctypedef Array4NonOwn[double] F64View4D
ctypedef Array5NonOwn[double] F64View5D
ctypedef Array1Own[double] F64Arr
ctypedef Array1Own[double] F64Arr1D
ctypedef Array2Own[double] F64Arr2D
ctypedef Array3Own[double] F64Arr3D
ctypedef Array4Own[double] F64Arr4D
ctypedef Array5Own[double] F64Arr5D
ctypedef double f64