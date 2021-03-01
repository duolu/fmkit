
#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <math.h>


#define MAX_L1 2048
#define MAX_L2 2048

static float cost(float *a, float *b, int d) {

	double ret = 0;

	for (int i = 0; i < d; i++) {

		ret += (a[i] - b[i]) * (a[i] - b[i]);

	}

	ret = (float)sqrt(ret);
	return ret;
}


static float dtw(float **data1_ptr, float **data2_ptr, int d, int l1, int l2, 
	int s1, int s2, int s_dists, int s_dir, int s_aligned, int window, float penalty,
	float **dists_ptr, int **dir_ptr, 
	int *a1start, int *a1end, int *a2start, int *a2end, 
	float **data2_aligned_ptr) {

	// CAUTION: Stride is not necessary n * sizeof(float). The internal
	// memory model of ndarray may require padding or append arbitrary 
	// padding. Thus, always use stride.

	float (*data1)[s1] = (float (*)[s1])data1_ptr;
	float (*data2)[s2] = (float (*)[s2])data2_ptr;

	float (*dists)[s_dists] = (float (*)[s_dists])dists_ptr;
	int (*dir)[s_dir] = (int (*)[s_dir])dir_ptr;

	float (*data2_aligned)[s_aligned] = (float (*)[s_aligned])data2_aligned_ptr;

//	int min_l = l1 < l2 ? l1 : l2;
//	for (int i = 0; i < min_l; i++) {
//
//		printf("%.6f, %.6f\n", data1[i][0], data2[i][0]);
//
//	}


	for (int i = 0; i < l1 + 1; i++) {

		for (int j = 0; j < l2 + 1; j++) {

			dists[i][j] = 1e9;
			dir[i][j] = 0;
		}

	}

	dists[0][0] = 0;
	dir[0][0] = 0;

	if (window < 0)
		window = l2 * 2;

	// CAUTION: both i and j start from 1
	for (int i = 1; i < l1 + 1; i++) {

		int jnew = (int)((float) l2 / l1 * i);

		int start = (jnew - window > 1) ? (jnew - window) : 1;
		int end = (jnew + window < l2 + 1) ? (jnew + window) : l2 + 1;


		for (int j = start; j < end; j++) {

            		// CAUTION: data1[0] and data2[0] mapps to dists[1][1],
            		//          and i, j here are indexing dists instead of data1 or data2,
            		//          i.e., dists[i][j] is comparing data1[i - 1] and data2[j - 1]
			float c = cost(data1[i - 1], data2[j - 1], d);


			float min = dists[i - 1][j - 1];
			dir[i][j] = 1; // 1 stands for diagonal

			if (dists[i - 1][j] + penalty < min) {
				min = dists[i - 1][j] + penalty;
				dir[i][j] = 2; // 2 stands for the i direction
			}

			if (dists[i][j - 1] + penalty < min) {
				min = dists[i][j - 1] + penalty;
				dir[i][j] = 4; // 4 stands for the j direction
			}

			dists[i][j] = c + min;


		}

	}


	// trace back the warping path to find element-wise mapping,
	// i.e., find a1start, a1end, a2start, a2end

	a1start[l1 - 1] = l2 - 1;
	a1end[l1 - 1] = l2 - 1;
	a2start[l2 - 1] = l1 - 1;
	a2end[l2 - 1] = l1 - 1;

	int i = l1;
	int j = l2;
	while (1) {

		if (dir[i][j] == 2) { // the i direction

			i -= 1;
		                        
			a1start[i - 1] = j - 1;
			a1end[i - 1] = j - 1;
			a2start[j - 1] = i - 1;

		} else if (dir[i][j] == 4) { // the j direction

			j -= 1;
		        
			a2start[j - 1] = i - 1;
			a2end[j - 1] = i - 1;
			a1start[i - 1] = j - 1;

		} else if (dir[i][j] == 1) { // the diagonal direction

			i -= 1;
			j -= 1;
			if (i == 0 && j == 0)
				break;
		    
			a1start[i - 1] = j - 1;
			a1end[i - 1] = j - 1;
			a2start[j - 1] = i - 1;
			a2end[j - 1] = i - 1;
		
		} else // dir[i][j] == 0, the corner
			break;

	}


	
	for (int i = 0; i < l1; i++) {
		for (int j = 0; j < d; j++) {

			if (a1start[i] == a1end[i]) {
				data2_aligned[i][j] = data2[a1start[i]][j];
			} else {

				float sum = 0;

				for (int k = a1start[i]; k <= a1end[i]; k++) {
					sum += data2[k][j];
				}
				sum /= (a1end[i] - a1start[i] + 1);
				data2_aligned[i][j] = sum;
			}

		}
	}


	return dists[l1][l2];
}



// warper

static PyObject *fmkit_utilities_dtw( PyObject *self, PyObject *args ) {

	PyObject *arg1 = NULL;
	PyObject *arg2 = NULL;
	int d;
	int l1;
	int l2;
	int window;
	float penalty;
	PyObject *arg_dists = NULL;
	PyObject *arg_dir = NULL;
	PyObject *arg_a1start = NULL;
	PyObject *arg_a1end = NULL;
	PyObject *arg_a2start = NULL;
	PyObject *arg_a2end = NULL;
	PyObject *arg_data2_aligned = NULL;
	//npy_intp matrix_dims[2];
	//npy_intp l1_dims[1];
	//npy_intp l2_dims[1];
	//npy_intp aligned_dims[2];

	//printf("fmkit_utilities_dtw() start.\n");

	if (!PyArg_ParseTuple(args, "OOiiiifOOOOOOO", &arg1, &arg2, &d, &l1, &l2, &window, &penalty,
		&arg_dists, &arg_dir, &arg_a1start, &arg_a1end, &arg_a2start, &arg_a2end, &arg_data2_aligned)) {

		//printf("fmkit_utilities_dtw() invalid arguments!!!");
		return NULL;
	}

	// printf("arguments: %p, %p, %d, %d, %d, %d, %f.\n", arg1, arg2, d, l1, l2, window, penalty);


	// prepare argument


	PyArrayObject *np_data1 = (PyArrayObject *)arg1;
	PyArrayObject *np_data2 = (PyArrayObject *)arg2;

	PyArrayObject *np_dists = (PyArrayObject *)arg_dists;
	PyArrayObject *np_dir = (PyArrayObject *)arg_dir;

	PyArrayObject *np_a1start = (PyArrayObject *)arg_a1start;
	PyArrayObject *np_a1end = (PyArrayObject *)arg_a1end;

	PyArrayObject *np_a2start = (PyArrayObject *)arg_a2start;
	PyArrayObject *np_a2end = (PyArrayObject *)arg_a2end;

	PyArrayObject *np_data2_aligned = (PyArrayObject *)arg_data2_aligned;

	//printf("fmkit_utilities_dtw() array parsing finishes.\n");
/*
	matrix_dims[0] = l1 + 1;
	matrix_dims[1] = l2 + 1;
	PyArrayObject *np_dists = (PyArrayObject *)PyArray_SimpleNew(2, matrix_dims, NPY_FLOAT);
	PyArrayObject *np_dir = (PyArrayObject *)PyArray_SimpleNew(2, matrix_dims, NPY_INT);

	//printf("1 ");

	l1_dims[0] = l1;
	PyArrayObject *np_a1start = (PyArrayObject *)PyArray_SimpleNew(1, l1_dims, NPY_INT);
	PyArrayObject *np_a1end = (PyArrayObject *)PyArray_SimpleNew(1, l1_dims, NPY_INT);

	//printf("2 ");

	l2_dims[0] = l2;
	PyArrayObject *np_a2start = (PyArrayObject *)PyArray_SimpleNew(1, l2_dims, NPY_INT);
	PyArrayObject *np_a2end = (PyArrayObject *)PyArray_SimpleNew(1, l2_dims, NPY_INT);

	//printf("3 ");

	aligned_dims[0] = l1;
	aligned_dims[1] = d;
	PyArrayObject *np_data2_aligned = (PyArrayObject *)PyArray_SimpleNew(2, aligned_dims, NPY_FLOAT);

	//printf("\n");

	//printf("fmkit_utilities_dtw() memory allocation finishes.\n");
*/

	float **data1 = (float **)PyArray_DATA(np_data1);
	float **data2 = (float **)PyArray_DATA(np_data2);

	float **dists = (float **)PyArray_DATA(np_dists);
	int **dir = (int **)PyArray_DATA(np_dir);

	int *a1start = (int *)PyArray_DATA(np_a1start);
	int *a1end = (int *)PyArray_DATA(np_a1end);
	int *a2start = (int *)PyArray_DATA(np_a2start);
	int *a2end = (int *)PyArray_DATA(np_a2end);

	float **data2_aligned = (float **)PyArray_DATA(np_data2_aligned);
/*
	int nd1 = (int)PyArray_NDIM(np_data1);
	int nd2 = (int)PyArray_NDIM(np_data2);
	int nd_dist = (int)PyArray_NDIM(np_dists);
	int nd_dir = (int)PyArray_NDIM(np_dir);

	printf("%d, %d, %d, %d\n", nd1, nd2, nd_dist, nd_dir);
*/
	npy_intp *ps1 = PyArray_STRIDES(np_data1);
	npy_intp *ps2 = PyArray_STRIDES(np_data2);
	npy_intp *ps_dists = PyArray_STRIDES(np_dists);
	npy_intp *ps_dir = PyArray_STRIDES(np_dir);
	npy_intp *ps_aligned = PyArray_STRIDES(np_data2_aligned);

	// printf("%d, %d, %d, %d\n", ps1[0], ps2[0], ps_dists[0], ps_dir[0]);


	//printf("fmkit_utilities_dtw() argument ready.\n");

	// do the actual computation

	float dist = dtw(data1, data2, d, l1, l2, 
		ps1[0] / sizeof(float), ps2[0] / sizeof(float), 
		ps_dists[0] / sizeof(float), ps_dir[0] / sizeof(float), 
		ps_aligned[0] / sizeof(float),
		window, penalty, dists, dir, 
		a1start, a1end, a2start, a2end, data2_aligned);

	//printf("fmkit_utilities_dtw() results ready.\n");

	// retrieve results

/*
	PyObject *tup = PyTuple_New(8);
	PyTuple_SetItem(tup, 0, Py_BuildValue("f", dist));
	PyTuple_SetItem(tup, 1, (PyObject *)np_dists);
	PyTuple_SetItem(tup, 2, (PyObject *)np_dir);
	PyTuple_SetItem(tup, 3, (PyObject *)np_a1start);
	PyTuple_SetItem(tup, 4, (PyObject *)np_a1end);
	PyTuple_SetItem(tup, 5, (PyObject *)np_a2start);
	PyTuple_SetItem(tup, 6, (PyObject *)np_a2end);
	PyTuple_SetItem(tup, 7, (PyObject *)np_data2_aligned);
*/

	//Py_DECREF(np_data1);
	//Py_DECREF(np_data2);
    

	return Py_BuildValue("f", dist);

}

static char fmkit_utilities_docs[] =
	"dtw_c(): Dynamic Time Warping implemented in C\n";

static PyMethodDef fmkit_utilities_methods[] = {
	{"dtw_c",  fmkit_utilities_dtw, METH_VARARGS, fmkit_utilities_docs},
	{NULL, NULL, 0, NULL}        // Sentinel
};

//PyMODINIT_FUNC init_utilities()  {
//	Py_InitModule3("utilities", utilities_methods);
//	//import_array();  // Must be present for NumPy.  Called first after above line.
//}

static struct PyModuleDef fmkit_utilities =
{
    PyModuleDef_HEAD_INIT,
    "fmkit_utilities", /* name of module */
    "Utility functions of fmkit package\n", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    fmkit_utilities_methods
};

PyMODINIT_FUNC PyInit_fmkit_utilities(void)
{
	import_array();  // Must be present for NumPy.  Called first after above line.
    return PyModule_Create(&fmkit_utilities);
}


