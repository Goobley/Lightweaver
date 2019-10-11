#ifndef CMO_ARRAY_HPP
#define CMO_ARRAY_HPP

#include <vector>
#include <array>
#include <cstdio>
#include <cassert>
#include <utility>

namespace Jasnah
{
#ifdef CMO_ARRAY_BOUNDS_CHECK
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim0))
    #define DO_BOUNDS_CHECK_M1()
#else
    #define DO_BOUNDS_CHECK()
    #define DO_BOUNDS_CHECK_M1()
#endif

template <typename T>
struct Array1NonOwn
{
    T* data;
    size_t Ndim;
    size_t dim0;
    Array1NonOwn() : data(nullptr), Ndim(1), dim0(0)
    {}
    Array1NonOwn(T* data_, size_t dim) : data(data_), Ndim(1), dim0(dim)
    {}
    Array1NonOwn(const Array1NonOwn& other) = default;
    Array1NonOwn(Array1NonOwn&& other) = default;

    Array1NonOwn&
    operator=(const Array1NonOwn& other) = default;

    Array1NonOwn&
    operator=(Array1NonOwn&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dim0; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dim0; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return &dim0;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim0;
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0)
    {
        DO_BOUNDS_CHECK();
        return data[i0];
    }
    inline T operator()(size_t i0) const
    {
        DO_BOUNDS_CHECK();
        return data[i0];
    }
};

template <typename T>
struct Array1Own
{
    std::vector<T> data;
    size_t Ndim;
    size_t dim0;
    Array1Own(size_t size) : data(size), Ndim(1), dim0(size)
    {}
    Array1Own(T val, size_t size) : data(size, val), Ndim(1), dim0(size)
    {}
    Array1Own(Array1NonOwn<T> other) : Ndim(other.Ndim), dim0(other.dim0), data(other.data, other.data + other.dim0)
    {}
    Array1Own(const Array1NonOwn<T>& other) : Ndim(other.Ndim), dim0(other.dim0), data(other.data, other.data + other.dim0)
    {}
    Array1Own(const Array1Own& other) = default;
    Array1Own(Array1Own&& other) = default;

    Array1Own&
    operator=(const Array1NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim0 = other.dim0;
        data.assign(other.data, other.data + other.dim0);
        return *this;
    }
    Array1Own&
    operator=(const Array1Own& other) = default;

    Array1Own&
    operator=(Array1Own&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dim0; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dim0; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return &dim0;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim0;
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0)
    {
        DO_BOUNDS_CHECK();
        return data[i0];
    }
    inline T operator()(size_t i0) const
    {
        DO_BOUNDS_CHECK();
        return data[i0];
    }
};

#if defined(DO_BOUNDS_CHECK) && defined(ARRAY_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]))
#endif

template <typename T>
struct Array2NonOwn
{
    T* data;
    size_t Ndim;
    std::array<const size_t, 2> dim;
    Array2NonOwn() : data(nullptr), Ndim(2), dim{}
    {}
    Array2NonOwn(T* data_, size_t dim0, size_t dim1) : data(data_), Ndim(2), dim{dim0, dim1}
    {}
    Array2NonOwn(const Array2NonOwn& other) = default;
    Array2NonOwn(Array2NonOwn&& other) = default;

    Array2NonOwn&
    operator=(const Array2NonOwn& other) = default;

    Array2NonOwn&
    operator=(Array2NonOwn&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dim[0]*dim[1]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dim[0]*dim[1]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dim[1] + i1];
    }
    inline T operator()(size_t i0, size_t i1) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dim[1] + i1];
    }
    const T* const_slice(size_t i0) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dim[1]];
    }

    Array1NonOwn<T> operator()(size_t i0)
    {
        return Array1NonOwn<T>(data[i0*dim[1]], dim[1]);
    }

    Array1NonOwn<const T> operator()(size_t i0) const
    {
        return Array1NonOwn<const T>(const_slice(i0), dim[1]);
    }
};

template <typename T>
struct Array2Own
{
    std::vector<T> data;
    size_t Ndim;
    std::array<const size_t, 2> dim;
    Array2Own(size_t size1, size_t size2) : data(size1*size2), Ndim(2), dim{size1, size2}
    {}
    Array2Own(T val, size_t size1, size_t size2) : data(size1*size2, val), Ndim(2), dim{size1, size2}
    {}
    Array2Own(Array2NonOwn<T> other) : Ndim(other.Ndim), dim(other.dim), data(other.data, other.data+other.dim0)
    {}
    Array2Own(const Array2NonOwn<T>& other) : Ndim(other.Ndim), dim(other.dim), data(other.data, other.data+other.dim0)
    {}
    Array2Own(const Array2Own& other) = default;
    Array2Own(Array2Own&& other) = default;
    
    Array2Own&
    operator=(const Array2NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        auto len = dim[0]*dim[1];
        data.assign(other.data, other.data+len);
        return *this;
    }

    Array2Own&
    operator=(const Array2Own& other) = default;

    Array2Own&
    operator=(Array2Own&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dim[0]*dim[1]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dim[0]*dim[1]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dim[1] + i1];
    }
    inline T operator()(size_t i0, size_t i1) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dim[1] + i1];
    }
    const T* const_slice(size_t i0) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dim[1]];
    }

    Array1NonOwn<T> operator()(size_t i0)
    {
        return Array1NonOwn<T>(data[i0*dim[1]], dim[1]);
    }

    Array1NonOwn<const T> operator()(size_t i0) const
    {
        return Array1NonOwn<const T>(const_slice(i0), dim[1]);
    }
};

#if defined(DO_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]))
#endif

template <typename T>
struct Array3NonOwn
{
    T* data;
    size_t Ndim;
    std::array<const size_t, 3> dim;
    std::array<const size_t, 2> dimProd;
    Array3NonOwn() : data(nullptr), Ndim(3), dim{0}, dimProd{0}
    { 
    }
    Array3NonOwn(T* data_, size_t dim0, size_t dim1, size_t dim2) : data(data_), Ndim(3), dim{dim0, dim1, dim2}, dimProd{dim1*dim2, dim2}
    {}
    Array3NonOwn(const Array3NonOwn& other) = default;
    Array3NonOwn(Array3NonOwn&& other) = default;

    Array3NonOwn&
    operator=(const Array3NonOwn& other) = default;

    Array3NonOwn&
    operator=(Array3NonOwn&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2];
    }
    inline T operator()(size_t i0, size_t i1, size_t i2) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2];
    }

    const T* const_slice(size_t i0, size_t i1) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1]];
    }

    Array2NonOwn<T> operator()(size_t i0)
    {
        return Array2NonOwn<T>(&(*this)(i0, 0, 0), dim[1], dim[2]);
    }

    Array2NonOwn<const T> operator()(size_t i0) const
    {
        return Array2NonOwn<const T>(const_slice(i0, 0), dim[1], dim[2]);
    }

    Array1NonOwn<T> operator()(size_t i0, size_t i1)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, 0), dim[2]);
    }

    Array1NonOwn<const T> operator()(size_t i0, size_t i1) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1), dim[2]);
    }
};

template <typename T>
struct Array3Own
{
    std::vector<T> data;
    size_t Ndim;
    std::array<const size_t, 3> dim;
    std::array<const size_t, 2> dimProd;
    Array3Own(size_t dim0, size_t dim1, size_t dim2) : data(dim0*dim1*dim2), Ndim(3), dim{dim0, dim1, dim2}, dimProd{dim1*dim2, dim2}
    {}
    Array3Own(T val, size_t dim0, size_t dim1, size_t dim2) : data(dim0*dim1*dim2, val), Ndim(3), dim{dim0, dim1, dim2}, dimProd{dim1*dim2, dim2}
    {}
    Array3Own(Array3NonOwn<T> other) : Ndim(other.ndim), dim(other.dim), dimProd(other.dimProd), 
                                       data(other.data, other.data+other.dim[0]*other.dim[1]*other.dim[2])
    {}
    Array3Own(const Array3NonOwn<T>& other) : Ndim(other.ndim), dim(other.dim), dimProd(other.dimProd), 
                                              data(other.data, other.data+other.dim[0]*other.dim[1]*other.dim[2])
    {}
    Array3Own(const Array3Own& other) = default;
    Array3Own(Array3Own&& other) = default;

    Array3Own&
    operator=(const Array3NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        dimProd = other.dimProd;
        auto len = other.dim[0]*other.dim[1]*other.dim[2];
        data.assign(other.data, other.data+len);
        return *this;
    }

    Array3Own&
    operator=(const Array3Own& other) = default;

    Array3Own&
    operator=(Array3Own&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2];
    }
    inline T operator()(size_t i0, size_t i1, size_t i2) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2];
    }

    const T* const_slice(size_t i0, size_t i1) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1]];
    }

    Array2NonOwn<T> operator()(size_t i0)
    {
        return Array2NonOwn<T>(&(*this)(i0, 0, 0), dim[1], dim[2]);
    }

    Array2NonOwn<const T> operator()(size_t i0) const
    {
        return Array2NonOwn<const T>(const_slice(i0, 0), dim[1], dim[2]);
    }

    Array1NonOwn<T> operator()(size_t i0, size_t i1)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, 0), dim[2]);
    }

    Array1NonOwn<const T> operator()(size_t i0, size_t i1) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1), dim[2]);
    }
};

#if defined(DO_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]) && (i3 >= 0) && (i3 < dim[3]))
#endif

template <typename T>
struct Array4NonOwn
{
    T* data;
    size_t Ndim;
    std::array<size_t, 4> dim;
    std::array<size_t, 3> dimProd;
    Array4NonOwn() : data(nullptr), Ndim(4), dim{0}, dimProd{0}
    {}
    Array4NonOwn(T* data_, size_t dim0, size_t dim1, size_t dim2, size_t dim3) : data(data_), Ndim(4), dim{dim0, dim1, dim2, dim3}, 
                                                                                 dimProd{dim1*dim2*dim3, dim2*dim3, dim3}
    {}
    Array4NonOwn(const Array4NonOwn& other) = default;
    Array4NonOwn(Array4NonOwn&& other) = default;

    Array4NonOwn&
    operator=(const Array4NonOwn& other) = default;

    Array4NonOwn&
    operator=(Array4NonOwn&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2, size_t i3)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    inline T operator()(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    const T* const_slice(size_t i0, size_t i1, size_t i2) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2]];
    }

    Array3NonOwn<T> operator()(size_t i0)
    {
        return Array3NonOwn<T>(&(*this)(i0, 0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array3NonOwn<const T> operator()(size_t i0) const
    {
        return Array3NonOwn<T>(const_slice(i0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array2NonOwn<T> operator()(size_t i0, size_t i1)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, 0, 0), dim[2], dim[3]);
    }

    Array2NonOwn<const T> operator()(size_t i0, size_t i1) const
    {
        return Array2NonOwn<T>(const_slice(i0, i1, 0), dim[2], dim[3]);
    }

    Array1NonOwn<T> operator()(size_t i0, size_t i1, size_t i2)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, 0), dim[3]);
    }

    Array1NonOwn<const T> operator()(size_t i0, size_t i1, size_t i2) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2), dim[3]);
    }
};

template <typename T>
struct Array4Own
{
    std::vector<T> data;
    size_t Ndim;
    std::array<size_t, 4> dim;
    std::array<size_t, 3> dimProd;
    Array4Own(size_t dim0, size_t dim1, size_t dim2, size_t dim3) : data(dim0*dim1*dim2*dim3), Ndim(4), dim{dim0, dim1, dim2, dim3},
                                                                    dimProd{dim1*dim2*dim3, dim2*dim3, dim3}
    {}
    Array4Own(T val, size_t dim0, size_t dim1, size_t dim2, size_t dim3) : data(dim0*dim1*dim2*dim3, val), Ndim(4), dim{dim0, dim1, dim2, dim3}, 
                                                                              dimProd{dim1*dim2*dim3, dim2*dim3, dim3}
    {}
    Array4Own(Array4NonOwn<T> other) : data(other.data, other.data+other.dim0*other.dim1*other.dim2*other.dim3), 
                                       Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}
    Array4Own(const Array4NonOwn<T>& other) : data(other.data, other.data+other.dim0*other.dim1*other.dim2*other.dim3), 
                                              Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}

    Array4Own(const Array4Own& other) = default;
    Array4Own(Array4Own&& other) = default;

    Array4Own&
    operator=(const Array4NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        dimProd = other.dimProd;
        auto len = other.dimProd[0] * other.dim[0];
        data.assign(other.data, other.data+len);
        return *this;
    }
    Array4Own&
    operator=(const Array4Own& other) = default;

    Array4Own&
    operator=(Array4Own&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2, size_t i3)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    inline T operator()(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    const T* const_slice(size_t i0, size_t i1, size_t i2) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2]];
    }

    Array3NonOwn<T> operator()(size_t i0)
    {
        return Array3NonOwn<T>(&(*this)(i0, 0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array3NonOwn<const T> operator()(size_t i0) const
    {
        return Array3NonOwn<T>(const_slice(i0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array2NonOwn<T> operator()(size_t i0, size_t i1)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, 0, 0), dim[2], dim[3]);
    }

    Array2NonOwn<const T> operator()(size_t i0, size_t i1) const
    {
        return Array2NonOwn<T>(const_slice(i0, i1, 0), dim[2], dim[3]);
    }

    Array1NonOwn<T> operator()(size_t i0, size_t i1, size_t i2)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, 0), dim[3]);
    }

    Array1NonOwn<const T> operator()(size_t i0, size_t i1, size_t i2) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2), dim[3]);
    }
};

#if defined(DO_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]) && (i3 >= 0) && (i3 < dim[3]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]) && (i3 >= 0) && (i3 < dim[3]) && (i4 >= 0) && (i4 < dim[4]))
#endif

template <typename T>
struct Array5NonOwn
{
    T* data;
    size_t Ndim;
    std::array<size_t, 5> dim;
    std::array<size_t, 4> dimProd;
    Array5NonOwn() : data(nullptr), Ndim(5), dim{0}, dimProd{0}
    {}
    Array5NonOwn(T* data_, size_t dim0, size_t dim1, size_t dim2, size_t dim3, size_t dim4) : data(data_), Ndim(5), dim{dim0,dim1,dim2,dim3,dim4}, 
                                                                                              dimProd{dim1*dim2*dim3*dim4, dim2*dim3*dim4, dim3*dim4, dim4}
    {}
    Array5NonOwn(const Array5NonOwn& other) = default;
    Array5NonOwn(Array5NonOwn&& other) = default;

    Array5NonOwn&
    operator=(const Array5NonOwn& other) = default;

    Array5NonOwn&
    operator=(Array5NonOwn&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }
    inline T operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }

    const T* const_slice(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3]];
    }

    Array4NonOwn<T> operator()(size_t i0)
    {
        return Array4NonOwn<T>(&(*this)(i0, 0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }
    Array4NonOwn<const T> operator()(size_t i0) const
    {
        return Array4NonOwn<const T>(const_slice(i0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<T> operator()(size_t i0, size_t i1)
    {
        return Array3NonOwn<T>(&(*this)(i0, i1, 0, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<const T> operator()(size_t i0, size_t i1) const
    {
        return Array3NonOwn<const T>(const_slice(i0, i1, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array2NonOwn<T> operator()(size_t i0, size_t i1, size_t i2)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, i2, 0, 0), dim[3], dim[4]);
    }

    Array2NonOwn<const T> operator()(size_t i0, size_t i1, size_t i2) const
    {
        return Array2NonOwn<const T>(const_slice(i0, i1, i2, 0), dim[3], dim[4]);
    }

    Array1NonOwn<T> operator()(size_t i0, size_t i1, size_t i2, size_t i3)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, i3, 0), dim[4]);
    }

    Array1NonOwn<const T> operator()(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2, i3), dim[4]);
    }
};

template <typename T>
struct Array5Own
{
    std::vector<T> data;
    size_t Ndim;
    std::array<size_t, 5> dim;
    std::array<size_t, 4> dimProd;
    Array5Own(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) : data(d0*d1*d2*d3*d4), Ndim(5), dim{d0,d1,d2,d3,d4}, 
                                                                       dimProd{d1*d2*d3*d4, d2*d3*d4, d3*d4, d4}
    {}
    Array5Own(T val, size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) : data(d0*d1*d2*d3*d4, val), Ndim(5), dim{d0,d1,d2,d3,d4}, 
                                                                              dimProd{d1*d2*d3*d4, d2*d3*d4, d3*d4, d4}
    {}
    Array5Own(Array5NonOwn<T> other) : data(other.data, other.data+other.dim[0]*other.dimProd[0]), Ndim(other.Ndim), 
                                       dim(other.dim), dimProd(other.dimProd)
    {}
    Array5Own(const Array5NonOwn<T>& other) : data(other.data, other.data+other.dim[0]*other.dimProd[0]), Ndim(other.Ndim), 
                                              dim(other.dim), dimProd(other.dimProd)
    {}
    Array5Own(const Array5Own& other) = default;
    Array5Own(Array5Own&& other) = default;

    Array5Own&
    operator=(const Array5NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        dimProd = other.dimProd;
        data.assign(other.data, other.data+other.dim[0]*other.dimProd[0]);
        return *this;
    }

    Array5Own&
    operator=(const Array5Own& other) = default;

    Array5Own&
    operator=(Array5Own&& other) = default;

    void fill(const T& val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }
    void fill(T val)
    {
        for (int i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const size_t* shape() const
    {
        return dim;
    }
    inline size_t shape(int i) const
    {
        if (i >= Ndim)
        {
            printf("oh no.\n");
            assert(false);
        }
        return dim[i];
    }

    inline T& operator[](size_t i)
    {
        return data[i];
    }
    inline T operator[](size_t i) const
    {
        return data[i];
    }

    inline T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }
    inline T operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }

    const T* const_slice(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3]];
    }

    Array4NonOwn<T> operator()(size_t i0)
    {
        return Array4NonOwn<T>(&(*this)(i0, 0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }
    Array4NonOwn<const T> operator()(size_t i0) const
    {
        return Array4NonOwn<const T>(const_slice(i0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<T> operator()(size_t i0, size_t i1)
    {
        return Array3NonOwn<T>(&(*this)(i0, i1, 0, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<const T> operator()(size_t i0, size_t i1) const
    {
        return Array3NonOwn<const T>(const_slice(i0, i1, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array2NonOwn<T> operator()(size_t i0, size_t i1, size_t i2)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, i2, 0, 0), dim[3], dim[4]);
    }

    Array2NonOwn<const T> operator()(size_t i0, size_t i1, size_t i2) const
    {
        return Array2NonOwn<const T>(const_slice(i0, i1, i2, 0), dim[3], dim[4]);
    }

    Array1NonOwn<T> operator()(size_t i0, size_t i1, size_t i2, size_t i3)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, i3, 0), dim[4]);
    }

    Array1NonOwn<const T> operator()(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2, i3), dim[4]);
    }
};

}

#ifndef CMO_ARRAY_NO_TYPEDEF
typedef Jasnah::Array1NonOwn<double> F64View;
typedef Jasnah::Array1NonOwn<double> F64View1D;
typedef Jasnah::Array2NonOwn<double> F64View2D;
typedef Jasnah::Array3NonOwn<double> F64View3D;
typedef Jasnah::Array4NonOwn<double> F64View4D;
typedef Jasnah::Array5NonOwn<double> F64View5D;
typedef Jasnah::Array1Own<double> F64Arr;
typedef Jasnah::Array1Own<double> F64Arr1D;
typedef Jasnah::Array2Own<double> F64Arr2D;
typedef Jasnah::Array3Own<double> F64Arr3D;
typedef Jasnah::Array4Own<double> F64Arr4D;
typedef Jasnah::Array5Own<double> F64Arr5D;
#endif

#else
#endif