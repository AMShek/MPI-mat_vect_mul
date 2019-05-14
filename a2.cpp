#include <iostream>
#include <fstream>
#include <memory.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <pthread.h>

using namespace std;

struct element {
	int row;
	int col;
	double value;
	element(){}
	element(int r, int c, double v) {
		row = r; col = c; value = v;
	}
};

int comm_sz;
int my_rank;
int rank=60222;
element** matrix;
int* counts;//用来存储0号进程要向某个进程发送的数据量
int* displs;//从第几个数据开始发
MPI_Comm comm;
int thread_count;

//计算scatter函数分别要向各个进程发送多少数据量(counts)，从哪里开始发(displs)
void Init_counts_displs(int counts[], int displs[], int n) ; 
void * Send_and_free(void* rank);

int main() {
	int n=1260994;//矩阵A中非零元的个数为1260994
	int local_n;
	element* local_matrix;
	double start, finish;//clocks
	
	MPI_Init(NULL, NULL);
	comm=MPI_COMM_WORLD;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	start=MPI_Wtime();
	//register new MPI data structure "element"
	int blockLength[]={1,1,1};
	MPI::Datatype oldTypes[]={MPI_INT,MPI_INT,MPI_DOUBLE};
	MPI::Aint addressOffsets[]={0,sizeof(int),sizeof(int)*2};
	MPI::Datatype newType=MPI::Datatype::Create_struct(sizeof(blockLength)/sizeof(int),blockLength,addressOffsets,oldTypes);
	newType.Commit();

	
	int quotient = n/comm_sz;
	int remainder = n % comm_sz;
	if (my_rank < remainder) {
		local_n = quotient +1;
	} 
	else {
		local_n = quotient;
	}
	//cout<<"Process #"<<my_rank<<" local_n="<<local_n<<endl;
	local_matrix=(element*)malloc(sizeof(element)*local_n);
	
	//Barrier 1
	//Now all processes allocate space for local_matrix
	MPI_Barrier(comm);

	//Process #0 reads matrix A
	if (my_rank==0){
		long thread;
		pthread_t* thread_handles;
		thread_count=comm_sz-1;
		thread_handles=(pthread_t*)malloc(thread_count*sizeof(pthread_t));
		//cout<<"create threads successfully!"<<endl;//test
		ifstream mat("/home/ambershek/HPC/lab4/matrix.mtx");
		int row, col;
		double value;
		int index=0;
		string header;
		counts=(int*)malloc(sizeof(int)*comm_sz);
		displs=(int*)malloc(sizeof(int)*comm_sz);
		if (displs!=NULL&&counts!=NULL){
			Init_counts_displs(counts,displs,n);
		}
		matrix=(element**)malloc(sizeof(element*)*comm_sz);
		matrix[0]=NULL;
		for (int i=1; i<comm_sz; i++){
			matrix[i]=(element*)malloc(sizeof(element)*counts[i]);
			//cout<<"Malloc successfully!"<<endl;//test
		}
		getline(mat,header);//delete the header
		//cout<<"Process#0 can read matrix file."<<endl;//test
		while (!mat.eof()) {
			mat >> row>>col>>value;
			if (row > 0 && col > 0 && value != 0 && mat.is_open()) {
				element new_element=element(row,col,value);
				if (index<counts[0]){
					local_matrix[index]=new_element;
					//cout<<"process#0"<<endl;
				}
				else{ 
					//cout<<"else";
					for (int i=1; i<comm_sz; i++){
						if (displs[i]<index && index<displs[i]+counts[i]){
							matrix[i][index-displs[i]]=new_element;
							break;
						}
					}
				}
				index++;
				//cout<<"index="<<index<<endl; //test
			}
			mat.get();
			if (mat.peek() == EOF) break;
		}
		//cout<<"Process #0 finish reading matrix A!"<<endl;

		for (thread=0; thread<thread_count; thread++){
			pthread_create(&thread_handles[thread],NULL,Send_and_free,(void*)(thread+1));
		}
		for (thread=0; thread<thread_count; thread++){
			pthread_join(thread_handles[thread],NULL);
		}
		free(thread_handles);
	}
	
	else {
		MPI_Recv(local_matrix, local_n, newType, 0, 0, comm, MPI_STATUS_IGNORE);
	}
	
	MPI_Barrier(comm);//test
	free(matrix);
	//All processes read vector x parallelly
	ifstream vec("/home/ambershek/HPC/lab4/vector.mtx");
	string vec_header;
	int vec_row, vec_col;
	double vec_element;
	double *x;
	int index_vector=0;
	
	x=(double*)malloc(sizeof(double)*rank);
	getline(vec,vec_header);//delete the header
	while (index_vector<rank){
		vec>>vec_row>>vec_col>>vec_element;
		x[index_vector]=vec_element;
		index_vector++;
	}
	//cout<<"Last element in x="<<x[rank-1]<<endl;//test
	//cout<<"Process #"<<my_rank<<" finishes reading x!"<<endl;
	//cout<<"The number of non-zero elements in vector x="<<index_vector+1<<endl;//test
	
	//Barrier 1
	//Now all processes have vector x and a part of A
	//Time to do calculation!
	MPI_Barrier(comm);
	//free useless arrays
	
	if (my_rank==0){
		free(displs);
		free(counts);
	}
	
	for (int i=0; i<local_n; i++){
		x[local_matrix[i].col] *= local_matrix[i].value;
	}
	//cout<<"Finish multiplication"<<endl;
	
	//Barrier 2
	//Now all processes have finished multiplcation
	//Time to gather!
	MPI_Barrier(comm);
	finish=MPI_Wtime();
	double* y;
	if (my_rank==0){
		y=(double*)malloc(sizeof(double)*rank);
		MPI_Reduce(&x[0], &y[0], rank, MPI_DOUBLE, MPI_SUM, 0, comm);
		finish=MPI_Wtime();
		cout<<"TIme elapsed: "<<finish-start<<" seconds"<<endl;
		ofstream out("/home/ambershek/HPC/lab4/result.txt");
		if (out.is_open()){
			for (int i=0; i<rank; i++){
				out<<y[i]<<"\n";
			}
			//cout<<"Finish outputing result!"<<endl;
		}
		free(y);	
	}
	else{
		MPI_Reduce(&x[0], &y[0], rank, MPI_DOUBLE, MPI_SUM, 0, comm);
	}
	MPI_Barrier(comm);
	free(x);
	free(local_matrix);
	newType.Free();
   	MPI_Finalize();
   return 0;
}

void Init_counts_displs(int counts[], int displs[], int n) {
	//count
   int offset, q, quotient, remainder;
   
   quotient = n/comm_sz;
   remainder = n % comm_sz;
   offset = 0;
   for (q = 0; q < comm_sz; q++) {
      if (q < remainder) 
         counts[q] = quotient+1;
      else 
         counts[q] = quotient;
      displs[q] = offset;
      offset += counts[q];
	  //cout<<"counts "<<q<<" = "<<counts[q]<<endl;//test
	  //cout<<"displs "<<q<<" = "<<displs[q]<<endl;//test
   }
	return ;
} 

void * Send_and_free(void* rank){
	long my_rank=(long) rank;
	int blockLength[]={1,1,1};
	MPI::Datatype oldTypes[]={MPI_INT,MPI_INT,MPI_DOUBLE};
	MPI::Aint addressOffsets[]={0,sizeof(int),sizeof(int)*2};
	MPI::Datatype newType=MPI::Datatype::Create_struct(sizeof(blockLength)/sizeof(int),blockLength,addressOffsets,oldTypes);
	newType.Commit();
	//cout<<"Thread #"<<my_rank<<" in Process #0"<<endl;
	MPI_Send(matrix[my_rank], counts[my_rank], newType, my_rank, 0, comm);
	free(matrix[my_rank]);
	newType.Free();
	return NULL;
}
