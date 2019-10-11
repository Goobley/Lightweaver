import numpy as np
cimport numpy as np

cdef extern from "CmoArray.hpp" namespace "Jasnah":
    cdef cppclass Array1NonOwn[T]:
        Array1NonOwn()
        Array1NonOwn(T*, size_t)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array1Own[T]:
        Array1Own(size_t)
        Array1Own(T, size_t)
        Array1Own(Array1NonOwn)
        Array1Own(const Array1NonOwn&)
        Array1Own& operator=(const Array1NonOwn&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array2NonOwn[T]:
        Array2NonOwn()
        Array2NonOwn(T*, size_t, size_t)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t) const
        T& operator()(size_t, size_t)
        Array1NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array2Own[T]:
        Array2Own(size_t, size_t)
        Array2Own(T, size_t, size_t)
        Array2Own(Array2NonOwn)
        Array2Own(const Array2NonOwn&)
        Array2Own& operator=(const Array2NonOwn&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array3NonOwn[T]:
        Array3NonOwn()
        Array3NonOwn(T*, size_t, size_t, size_t)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t, size_t) const
        T& operator()(size_t, size_t, size_t)
        Array1NonOwn[T] operator()(size_t, size_t)
        Array2NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array3Own[T]:
        Array3Own(size_t, size_t, size_t)
        Array3Own(T, size_t, size_t, size_t)
        Array3Own(Array3NonOwn)
        Array3Own(const Array3NonOwn&)
        Array3Own& operator=(const Array3NonOwn&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t, size_t)
        Array2NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array4NonOwn[T]:
        Array4NonOwn()
        Array4NonOwn(T*, size_t, size_t, size_t, size_t)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t, size_t, size_t) const
        T& operator()(size_t, size_t, size_t, size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t)
        Array3NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array4Own[T]:
        Array4Own(size_t, size_t, size_t, size_t)
        Array4Own(T, size_t, size_t, size_t, size_t)
        Array4Own(Array4NonOwn)
        Array4Own(const Array4NonOwn&)
        Array4Own& operator=(const Array4NonOwn&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t)
        Array3NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array5NonOwn[T]:
        Array5NonOwn()
        Array5NonOwn(T*, size_t, size_t, size_t, size_t, size_t)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t, size_t, size_t, size_t, size_t) const
        T& operator()(size_t, size_t, size_t, size_t, size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t, size_t)
        Array3NonOwn[T] operator()(size_t, size_t)
        Array4NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

    cdef cppclass Array5Own[T]:
        Array5Own(size_t, size_t, size_t, size_t, size_t)
        Array5Own(T, size_t, size_t, size_t, size_t, size_t)
        Array5Own(Array5NonOwn)
        Array5Own(const Array5NonOwn&)
        Array5Own& operator=(const Array5NonOwn&)
        T operator[](size_t) const
        T& operator[](size_t)
        T operator()(size_t) const
        T& operator()(size_t)
        Array1NonOwn[T] operator()(size_t, size_t, size_t, size_t)
        Array2NonOwn[T] operator()(size_t, size_t, size_t)
        Array3NonOwn[T] operator()(size_t, size_t)
        Array4NonOwn[T] operator()(size_t)
        void fill(T)
        void fill(const T&)
        size_t* shape()
        size_t shape(size_t)

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