static PyObject *
PyUFunc_Accumulate(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *out,
                   int axis, int otype)
{
    PyArrayObject *op[2];
    PyArray_Descr *op_dtypes[2] = {NULL, NULL};
    int op_axes_arrays[2][NPY_MAXDIMS];
    int *op_axes[2] = {op_axes_arrays[0], op_axes_arrays[1]};
    npy_uint32 op_flags[2];
    int idim, ndim, otype_final;
    int needs_api, need_outer_iterator;

    NpyIter *iter = NULL, *iter_inner = NULL;

    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;

    const char *ufunc_name = ufunc->name ? ufunc->name : "(unknown)";

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_BEGIN_THREADS_DEF;

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s.accumulate\n", ufunc_name);

#if 0
    printf("Doing %s.accumulate on array with dtype :  ", ufunc_name);
    PyObject_Print((PyObject *)PyArray_DESCR(arr), stdout, 0);
    printf("\n");
#endif

    if (_get_bufsize_errmask(NULL, "accumulate", &buffersize, &errormask) < 0) {
        return NULL;
    }

    /* Take a reference to out for later returning */
    Py_XINCREF(out);

    otype_final = otype;
    if (get_binary_op_function(ufunc, &otype_final,
                                &innerloop, &innerloopdata) < 0) {
        PyArray_Descr *dtype = PyArray_DescrFromType(otype);
        PyErr_Format(PyExc_ValueError,
                     "could not find a matching type for %s.accumulate, "
                     "requested type has type code '%c'",
                            ufunc_name, dtype ? dtype->type : '-');
        Py_XDECREF(dtype);
        goto fail;
    }

    ndim = PyArray_NDIM(arr);

    /*
     * Set up the output data type, using the input's exact
     * data type if the type number didn't change to preserve
     * metadata
     */
    if (PyArray_DESCR(arr)->type_num == otype_final) {
        if (PyArray_ISNBO(PyArray_DESCR(arr)->byteorder)) {
            op_dtypes[0] = PyArray_DESCR(arr);
            Py_INCREF(op_dtypes[0]);
        }
        else {
            op_dtypes[0] = PyArray_DescrNewByteorder(PyArray_DESCR(arr),
                                                    NPY_NATIVE);
        }
    }
    else {
        op_dtypes[0] = PyArray_DescrFromType(otype_final);
    }
    if (op_dtypes[0] == NULL) {
        goto fail;
    }

#if NPY_UF_DBG_TRACING
    printf("Found %s.accumulate inner loop with dtype :  ", ufunc_name);
    PyObject_Print((PyObject *)op_dtypes[0], stdout, 0);
    printf("\n");
#endif

    /* Set up the op_axes for the outer loop */
    for (idim = 0; idim < ndim; ++idim) {
        op_axes_arrays[0][idim] = idim;
        op_axes_arrays[1][idim] = idim;
    }

    /* The per-operand flags for the outer loop */
    op_flags[0] = NPY_ITER_READWRITE |
                  NPY_ITER_NO_BROADCAST |
                  NPY_ITER_ALLOCATE |
                  NPY_ITER_NO_SUBTYPE;
    op_flags[1] = NPY_ITER_READONLY;

    op[0] = out;
    op[1] = arr;

    need_outer_iterator = (ndim > 1);
    /* We can't buffer, so must do UPDATEIFCOPY */
    if (!PyArray_ISALIGNED(arr) || (out && !PyArray_ISALIGNED(out)) ||
            !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(arr)) ||
            (out &&
             !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(out)))) {
        need_outer_iterator = 1;
    }

    if (need_outer_iterator) {
        int ndim_iter = 0;
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK|
                           NPY_ITER_REFS_OK;
        PyArray_Descr **op_dtypes_param = NULL;

        /*
         * The way accumulate is set up, we can't do buffering,
         * so make a copy instead when necessary.
         */
        ndim_iter = ndim;
        flags |= NPY_ITER_MULTI_INDEX;
        /* Add some more flags */
        op_flags[0] |= NPY_ITER_UPDATEIFCOPY|NPY_ITER_ALIGNED;
        op_flags[1] |= NPY_ITER_COPY|NPY_ITER_ALIGNED;
        op_dtypes_param = op_dtypes;
        op_dtypes[1] = op_dtypes[0];
        NPY_UF_DBG_PRINT("Allocating outer iterator\n");
        iter = NpyIter_AdvancedNew(2, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags,
                                   op_dtypes_param,
                                   ndim_iter, op_axes, NULL, 0);
        if (iter == NULL) {
            goto fail;
        }

        /* In case COPY or UPDATEIFCOPY occurred */
        op[0] = NpyIter_GetOperandArray(iter)[0];
        op[1] = NpyIter_GetOperandArray(iter)[1];

        if (PyArray_SIZE(op[0]) == 0) {
            if (out == NULL) {
                out = op[0];
                Py_INCREF(out);
            }
            goto finish;
        }

        if (NpyIter_RemoveAxis(iter, axis) != NPY_SUCCEED) {
            goto fail;
        }
        if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
            goto fail;
        }
    }

    /* Get the output */
    if (out == NULL) {
        if (iter) {
            op[0] = out = NpyIter_GetOperandArray(iter)[0];
            Py_INCREF(out);
        }
        else {
            PyArray_Descr *dtype = op_dtypes[0];
            Py_INCREF(dtype);
            op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                    &PyArray_Type, dtype,
                                    ndim, PyArray_DIMS(op[1]), NULL, NULL,
                                    0, NULL);
            if (out == NULL) {
                goto fail;
            }

        }
    }

    /*
     * If the reduction axis has size zero, either return the reduction
     * unit for UFUNC_REDUCE, or return the zero-sized output array
     * for UFUNC_ACCUMULATE.
     */
    if (PyArray_DIM(op[1], axis) == 0) {
        goto finish;
    }
    else if (PyArray_SIZE(op[0]) == 0) {
        goto finish;
    }

    if (iter && NpyIter_GetIterSize(iter) != 0) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];
        npy_intp count_m1, stride0, stride1;

        NpyIter_IterNextFunc *iternext;
        char **dataptr;

        int itemsize = op_dtypes[0]->elsize;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);


        /* Execute the loop with just the outer iterator */
        count_m1 = PyArray_DIM(op[1], axis)-1;
        stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with just outer iterator\n");

        stride0 = PyArray_STRIDE(op[0], axis);

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        needs_api = NpyIter_IterationNeedsAPI(iter);

        NPY_BEGIN_THREADS_NDITER(iter);

        do {
            dataptr_copy[0] = dataptr[0];
            dataptr_copy[1] = dataptr[1];
            dataptr_copy[2] = dataptr[0];

            /*
             * Copy the first element to start the reduction.
             *
             * Output (dataptr[0]) and input (dataptr[1]) may point to
             * the same memory, e.g. np.add.accumulate(a, out=a).
             */
            if (otype == NPY_OBJECT) {
                /*
                 * Incref before decref to avoid the possibility of the
                 * reference count being zero temporarily.
                 */
                Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                *(PyObject **)dataptr_copy[0] =
                                    *(PyObject **)dataptr_copy[1];
            }
            else {
                memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
            }

            if (count_m1 > 0) {
                /* Turn the two items into three for the inner loop */
                dataptr_copy[1] += stride1;
                dataptr_copy[2] += stride0;
                NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                (int)count_m1);
                innerloop(dataptr_copy, &count_m1,
                            stride_copy, innerloopdata);
            }
        } while (iternext(iter));

        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        int itemsize = op_dtypes[0]->elsize;

        /* Execute the loop with no iterators */
        npy_intp count = PyArray_DIM(op[1], axis);
        npy_intp stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with no iterators\n");

        if (PyArray_NDIM(op[0]) != PyArray_NDIM(op[1]) ||
                !PyArray_CompareLists(PyArray_DIMS(op[0]),
                                      PyArray_DIMS(op[1]),
                                      PyArray_NDIM(op[0]))) {
            PyErr_SetString(PyExc_ValueError,
                    "provided out is the wrong size "
                    "for the reduction");
            goto fail;
        }
        stride0 = PyArray_STRIDE(op[0], axis);

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        /* Turn the two items into three for the inner loop */
        dataptr_copy[0] = PyArray_BYTES(op[0]);
        dataptr_copy[1] = PyArray_BYTES(op[1]);
        dataptr_copy[2] = PyArray_BYTES(op[0]);

        /*
         * Copy the first element to start the reduction.
         *
         * Output (dataptr[0]) and input (dataptr[1]) may point to the
         * same memory, e.g. np.add.accumulate(a, out=a).
         */
        if (otype == NPY_OBJECT) {
            /*
             * Incref before decref to avoid the possibility of the
             * reference count being zero temporarily.
             */
            Py_XINCREF(*(PyObject **)dataptr_copy[1]);
            Py_XDECREF(*(PyObject **)dataptr_copy[0]);
            *(PyObject **)dataptr_copy[0] =
                                *(PyObject **)dataptr_copy[1];
        }
        else {
            memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
        }

        if (count > 1) {
            --count;
            dataptr_copy[1] += stride1;
            dataptr_copy[2] += stride0;

            NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)count);

            needs_api = PyDataType_REFCHK(op_dtypes[0]);

            if (!needs_api) {
                NPY_BEGIN_THREADS_THRESHOLDED(count);
            }

            innerloop(dataptr_copy, &count,
                        stride_copy, innerloopdata);

            NPY_END_THREADS;
        }
    }

finish:
    Py_XDECREF(op_dtypes[0]);
    NpyIter_Deallocate(iter);
    NpyIter_Deallocate(iter_inner);

    return (PyObject *)out;

fail:
    Py_XDECREF(out);
    Py_XDECREF(op_dtypes[0]);

    NpyIter_Deallocate(iter);
    NpyIter_Deallocate(iter_inner);

    return NULL;
}


