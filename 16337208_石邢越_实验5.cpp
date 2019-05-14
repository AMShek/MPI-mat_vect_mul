#include <iostream>
#include <math.h>
#include <mpi.h>
#include <fstream>
#include <memory.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdlib>
using namespace std;

int	my_rank, comm_sz;
int	row_comm_sz, col_comm_sz, diag_comm_sz;
int	my_row_rank, my_col_rank, my_diag_rank;
MPI_Comm comm, row_comm, col_comm, diag_comm;
int	diag, which_row_comm, which_col_comm;
MPI_Datatype submat_mpi_t;
int	*distrib_counts, *distrib_disps;

void Build_comms(void);
void Build_diag_comm(void);
void Check_for_error(int loc_ok, char fname[], char	message[]);
void Get_dims(int* m_p,	int* loc_m_p, int* n_p,	int* loc_n_p);
void Allocate_arrays(double** loc_A_pp,	double** loc_x_pp, 
	  double**	loc_y_pp, int m, int loc_m,	int	n, int loc_n);
void Init_distrib_data(int m, int loc_m, int n,	int	loc_n);
void Read_matrix(char prompt[],	double loc_A[],	int	m, int loc_m,
	  int n, int loc_n);
void Print_matrix(char title[],	double loc_A[],	int	m, int loc_m,
	  int n, int loc_n);
void Read_vector(char prompt[],	double loc_vec[], int n, int loc_n);
void Print_vector(char title[],	double loc_vec[], int n, int loc_n);
void Print_vector_file(char	title[], double	loc_vec[], int n, int loc_n);
void Mat_vect_mult(double loc_A[], double loc_x[], 
	  double loc_y[], int m, int loc_m, int n,	int	loc_n);
void Print_loc_vects(char title[], double loc_vec[], int loc_n);



int	main(void) {
   double* loc_A;
   double* loc_x;
   double* loc_y;
   
   int m,n;
   int loc_m, loc_n;
   double start, finish;//clocks
   MPI_Init(NULL, NULL);
   Build_comms();
   Build_diag_comm();
   
   Get_dims(&m,	&loc_m,	&n,	&loc_n);
	if (my_rank==0)start=MPI_Wtime();
   Allocate_arrays(&loc_A, &loc_x, &loc_y, m, loc_m, n,	loc_n);
   Init_distrib_data(m,	loc_m, n, loc_n);

   Read_matrix("A",	loc_A, m, loc_m, n,	loc_n);
   
   Read_vector("x",	loc_x, n, loc_n);
   
   Mat_vect_mult(loc_A,	loc_x, loc_y, m, loc_m,	n, loc_n);
   
   Print_vector_file("y", loc_y,	m, loc_m);
   
   free(loc_A);
   free(loc_x);
   free(loc_y);
   free(distrib_counts);
   free(distrib_disps);
   MPI_Type_free(&submat_mpi_t);
   MPI_Comm_free(&comm);
   MPI_Comm_free(&row_comm);
   MPI_Comm_free(&col_comm);
   if (diag) MPI_Comm_free(&diag_comm);
   if (my_rank==0){
	finish=MPI_Wtime();
   cout<<"TIme elapsed:	"<<finish-start<<" seconds"<<endl;
   }
   MPI_Finalize();
   return 0;
} 



void Check_for_error(
   int		 loc_ok	  /* in	*/,	
   char		 fname[]	 /* in */,
   char		 message[]	 /*	in */) {
   
   int ok;
   
   MPI_Allreduce(&loc_ok, &ok, 1, MPI_INT, MPI_MIN,	comm);
   if (ok == 0)	{
	  int my_rank;
	  MPI_Comm_rank(comm, &my_rank);
	  if (my_rank == 0) {
		 fprintf(stderr, "Proc %d	> In %s, %s\n",	my_rank, fname,	
				  message);
		 fflush(stderr);
	  }
	  MPI_Finalize();
	  exit(-1);
   }
}  



void Get_dims(
	  int*		 m_p		 /* out */, 
	  int*		 loc_m_p  /* out */,
	  int*		 n_p		 /* out */,
	  int*		 loc_n_p  /* out */) {
   
   int loc_ok =	1;
   
   if (my_rank == 0) {
	  printf("Enter the order of the matrix\n");
	  scanf("%d", m_p);
   }
   MPI_Bcast(m_p, 1, MPI_INT, 0, comm);
   *n_p	= *m_p;
   if (*m_p	<= 0 ||	*m_p % col_comm_sz != 0) loc_ok	= 0;
   Check_for_error(loc_ok, "Get_dims",
		 "order must be positive and evenly divisible	by sqrt(comm_sz)");

   *loc_m_p	= *m_p/col_comm_sz;
   *loc_n_p	= *n_p/row_comm_sz;
}  



void Allocate_arrays(
   double**	 loc_A_pp  /* out */, 
   double**	 loc_x_pp  /* out */, 
   double**	 loc_y_pp  /* out */, 
   int		 m			 /* in	 */,   
   int		 loc_m	   /* in	 */, 
   int		 n			 /* in	 */,
   int		 loc_n	   /* in	 */) {
   
   int loc_ok =	1;
   
   *loc_A_pp = (double*)malloc(loc_m*loc_n*sizeof(double));
   *loc_x_pp = (double*)malloc(loc_n*sizeof(double));
   *loc_y_pp = (double*)malloc(loc_m*sizeof(double));
   
   if (*loc_A_pp ==	NULL ||	loc_x_pp ==	NULL ||
	  loc_y_pp	== NULL) loc_ok	= 0;
   Check_for_error(loc_ok, "Allocate_arrays",
				  "Can't allocate local	arrays");
} 



void Build_comms(void) {
   MPI_Comm	grid_comm;
   int dim_sizes[2];
   int wrap_around[] = {0,0};
   int reorder = 1;
   int loc_ok =	1;
   int coords[2];
   int free_coords[2];

   comm	= MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   int q = sqrt(comm_sz);
   if (comm_sz != q*q) loc_ok =	0;
   Check_for_error(loc_ok, "Build_comms", "comm_sz not a perfect square");
   dim_sizes[0]	= dim_sizes[1] = q;
   MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder,
		 &grid_comm);
   comm	= grid_comm;
   MPI_Comm_size(grid_comm,	&comm_sz);
   MPI_Comm_rank(grid_comm,	&my_rank);

   free_coords[0] =	0;
   free_coords[1] =	1;
   MPI_Cart_sub(grid_comm, free_coords,	&row_comm);
   MPI_Comm_size(row_comm, &row_comm_sz);
   MPI_Comm_rank(row_comm, &my_row_rank);

   free_coords[0] =	1;
   free_coords[1] =	0;
   MPI_Cart_sub(grid_comm, free_coords,	&col_comm);
   MPI_Comm_size(col_comm, &col_comm_sz);
   MPI_Comm_rank(col_comm, &my_col_rank);

   if (my_row_rank == my_col_rank) 
	  diag	= 1;
   else
	  diag	= 0;
   MPI_Cart_coords(comm, my_rank, 2, coords);
   which_row_comm =	coords[0];
   which_col_comm =	coords[1];
}  



void Build_diag_comm(void) {
   int process_ranks[row_comm_sz], q;
   MPI_Group group,	diag_group;

   for (q =	0; q < row_comm_sz;	q++)
	  process_ranks[q]	= q*(row_comm_sz + 1);

   MPI_Comm_group(comm,	&group);

   MPI_Group_incl(group, row_comm_sz, process_ranks, &diag_group);

   MPI_Comm_create(comm, diag_group, &diag_comm);

   if (diag) {
	  MPI_Comm_size(diag_comm,	&diag_comm_sz);
	  MPI_Comm_rank(diag_comm,	&my_diag_rank);
   } else {
	  diag_comm_sz	= row_comm_sz;
	  my_diag_rank	= -1;
   }
}  



void Init_distrib_data(int m, int loc_m, int n,	int	loc_n) {
   int p_r,	p_c, proc;
   MPI_Datatype	vect_mpi_t;

   MPI_Type_vector(loc_m, loc_n, n,	MPI_DOUBLE,	&vect_mpi_t);
   MPI_Type_create_resized(vect_mpi_t, 0, loc_n*sizeof(double),	
		 &submat_mpi_t);
   MPI_Type_commit(&submat_mpi_t);
   MPI_Type_free(&vect_mpi_t);

   distrib_counts =	(int*)malloc(comm_sz*sizeof(int));
   distrib_disps = (int*)malloc(comm_sz*sizeof(int));

   for (p_r	= 0; p_r < col_comm_sz;	p_r++)
	  for (p_c	= 0; p_c < row_comm_sz;	p_c++) {
		 proc	= p_r*row_comm_sz +	p_c;
		 distrib_counts[proc]	= 1;		
		 distrib_disps[proc] = p_r*row_comm_sz*loc_m + p_c;
	  }
} 



void Read_matrix(
				char		  prompt[]	   /* in  */, 
				double		  loc_A[]		 /* out */, 
				int			  m			   /* in  */, 
				int			  loc_m		   /* in	 */, 
				int			  n			   /* in  */,
				int			  loc_n		   /* in	 */) {
   double* A = NULL;
   int loc_ok =	1;
   int i, j;

   if (my_rank == 0) {
	  A = (double*)malloc(m*n*sizeof(double));
	  if (A ==	NULL) loc_ok = 0;
	  Check_for_error(loc_ok, "Read_matrix",
				  "Can't allocate temporary	matrix");

	  printf("Calculating elements	of matrix %s...\n",	prompt);
	  for (i =	0; i < m; i++)
		 for (j =	0; j < n; j++)
			A[i*n+j]= i-	0.1*j +1;
	  
	  MPI_Scatterv(A, distrib_counts, distrib_disps, submat_mpi_t,
			loc_A, loc_m*loc_n, MPI_DOUBLE, 0, comm);
	  free(A);
   } else {
	  Check_for_error(loc_ok, "Read_matrix",
				  "Can't allocate temporary	matrix");
	  MPI_Scatterv(A, distrib_counts, distrib_disps, submat_mpi_t,
			loc_A, loc_m*loc_n, MPI_DOUBLE, 0, comm);
   }
}  



void Print_matrix(char title[],	double loc_A[],	int	m, int loc_m, 
	  int n, int loc_n) {
   double* A = NULL;
   int loc_ok =	1;
   int i, j;

   if (my_rank == 0) {
	  A = (double*)malloc(m*n*sizeof(double));
	  if (A ==	NULL) loc_ok = 0;
	  Check_for_error(loc_ok, "Print_matrix",
				  "Can't allocate temporary	matrix");

	  MPI_Gatherv(loc_A, loc_m*loc_n, MPI_DOUBLE, 
			A, distrib_counts, distrib_disps, submat_mpi_t,
			0, comm);

	  printf("The matrix %s\n", title);
	  for (i =	0; i < m; i++) {
		 for (j =	0; j < n; j++)
			printf("%.2f	", A[i*n+j]);
		 printf("\n");
	  }
	  
	  free(A);
   } else {
	  Check_for_error(loc_ok, "Print_matrix",
				  "Can't allocate temporary	matrix");
	  MPI_Gatherv(loc_A, loc_m*loc_n, MPI_DOUBLE, 
			A, distrib_counts, distrib_disps, submat_mpi_t,
			0, comm);
   }
}  



void Read_vector(
				char	  prompt[]		/*	in	*/, 
				double	  loc_vec[]	/* out */, 
				int		  n			/* in  */,
				int		  loc_n	  /* in  */) {
   double* vec = NULL;
   int i, loc_ok = 1;
   
   if (my_diag_rank	== 0) {
	  vec = (double*)malloc(n*sizeof(double));
	  if (vec == NULL)	loc_ok = 0;
	  Check_for_error(loc_ok, "Read_vector",
				  "Can't allocate temporary	vector");
	  printf("Calculating the elements	of vector %s...\n",	prompt);
	  for (i =	0; i < n; i++)
		 vec[i]=0.1*i;
	  MPI_Scatter(vec,	loc_n, MPI_DOUBLE,
			loc_vec,	loc_n, MPI_DOUBLE, 0, diag_comm);
	  free(vec);
   } else {
	  Check_for_error(loc_ok, "Read_vector",
				  "Can't allocate temporary	vector");
	  if (diag) 
		 MPI_Scatter(vec,	loc_n, MPI_DOUBLE,
				  loc_vec, loc_n, MPI_DOUBLE, 0, diag_comm);
   }
}  



void Mat_vect_mult(
	  double	 loc_A[]		/* in  */, 
	  double	 loc_x[]		/* in  */, 
	  double	 loc_y[]		/* out */,
	  int		 m			  /* in  */,
	  int		 loc_m		  /* in	 */, 
	  int		 n			  /* in  */,
	  int		 loc_n		  /* in	 */) {

   int loc_i, loc_j;
   double sub_y[loc_m];

   MPI_Bcast(loc_x,	loc_n, MPI_DOUBLE, which_col_comm, col_comm);

   for (loc_i =	0; loc_i < loc_m; loc_i++) {
	  sub_y[loc_i]	= 0.0;
	  for (loc_j =	0; loc_j < loc_n; loc_j++)
		 sub_y[loc_i]	+= loc_A[loc_i*loc_n + loc_j]*
			loc_x[loc_j];
   }

	*	store the result	in the diagonal	*/
   MPI_Reduce(sub_y, loc_y,	loc_m, MPI_DOUBLE, MPI_SUM,	
		 which_row_comm, row_comm);
} 



void Print_vector(
				 char	   title[]	 /* in	*/,	
				 double	   loc_vec[] /*	in */, 
				 int	   n			/* in */,
				 int	   loc_n	  /* in */) {
   double* vec = NULL;
   int i, loc_ok = 1;
   
   if (my_diag_rank	== 0) {
	  vec = (double*)malloc(n*sizeof(double));
	  if (vec == NULL)	loc_ok = 0;
	  Check_for_error(loc_ok, "Print_vector",
				  "Can't allocate temporary	vector");
	  MPI_Gather(loc_vec, loc_n, MPI_DOUBLE,
				  vec, loc_n, MPI_DOUBLE, 0, diag_comm);
	  printf("\nThe vector	%s\n", title);
	  for (i =	0; i < n; i++)
		 printf("%f ", vec[i]);
	  printf("\n");
	  free(vec);
   }  else {
	  Check_for_error(loc_ok, "Print_vector",
				  "Can't allocate temporary	vector");
	  if (diag)  
		 MPI_Gather(loc_vec, loc_n, MPI_DOUBLE,
							vec,	loc_n, MPI_DOUBLE, 0, diag_comm);
   }
}  



void Print_loc_vects(char title[], double loc_vec[], int loc_n)	{
   double temp_vect[loc_n];
   int q, i;

   if (my_rank == 0) {
	  printf("%s:\n", title);
	  printf("Proc	%d > ",	my_rank);
	  for (i =	0; i < loc_n; i++)
		 printf("%.2f	", loc_vec[i]);
	  printf("\n");
	  for (q =	1; q < comm_sz;	q++) {
		 MPI_Recv(temp_vect, loc_n, MPI_DOUBLE, q, 0,	comm,
				  MPI_STATUS_IGNORE);
		 printf("Proc	%d > ",	q);
		 for (i =	0; i < loc_n; i++)
			printf("%.2f	", temp_vect[i]);
		 printf("\n");
	  }
	  printf("\n");
   } else {
	  MPI_Send(loc_vec, loc_n,	MPI_DOUBLE,	0, 0, comm);
   }
}  

void Print_vector_file(
				 char	   title[]	 /* in	*/,	
				 double	   loc_vec[] /*	in */, 
				 int	   n			/* in */,
				 int	   loc_n	  /* in */) {
   double* vec = NULL;
   int i, loc_ok = 1;
   
   if (my_diag_rank	== 0) {
	  vec = (double*)malloc(n*sizeof(double));
	  if (vec == NULL)	loc_ok = 0;
	  Check_for_error(loc_ok, "Print_vector",
				  "Can't allocate temporary	vector");
	  MPI_Gather(loc_vec, loc_n, MPI_DOUBLE,
				  vec, loc_n, MPI_DOUBLE, 0, diag_comm);
	  printf("\nThe vector %s is outputing to a file...\n", title);

	ofstream out("/home/ambershek/HPC/lab5/result.txt");
	if	(out.is_open()){
		for (int i=0; i<n; i++){
			out<<vec[i]<<"\n";
		}
	}	   
	  free(vec);
   }  else {
	  Check_for_error(loc_ok, "Print_vector",
				  "Can't allocate temporary	vector");
	  if (diag)  
		 MPI_Gather(loc_vec, loc_n, MPI_DOUBLE,
							vec,	loc_n, MPI_DOUBLE, 0, diag_comm);
   }
}  
