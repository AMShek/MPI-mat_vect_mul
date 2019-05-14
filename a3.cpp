#include <iostream>
#include <fstream>
#include <memory.h>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

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
MPI_Comm comm;

//计算scatter函数分别要向各个进程发送多少数据量(counts)，从哪里开始发(displs)
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
} 

int main() {
	int n=1260994;//矩阵A中非零元的个数为1260994
	int local_n;
	element* local_matrix;
	element* matrix;
	int* counts;//用来存储0号进程要向某个进程发送的数据量
	int* displs;//从第几个数据开始发
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
	//Barrier1
	//Now processes besides process#0 has allocate space for subparts of matrix!
	MPI_Barrier(comm);
	
	//Process #0 reads matrix A
	if (my_rank==0){
		matrix=(element*)malloc(sizeof(element)*n);
		counts=(int*)malloc(sizeof(int)*comm_sz);
		displs=(int*)malloc(sizeof(int)*comm_sz);
		if (displs!=NULL&&counts!=NULL){
			//cout<<"n="<<n<<endl;//test
			Init_counts_displs(counts,displs,n);
		}
		ifstream mat("/home/ambershek/HPC/lab4/matrix.mtx");
		int row, col;
		double value;
		string header;
		int index_matrix=0;
		getline(mat,header);//delete the header
		//cout<<"Process#0 can read matrix file."<<endl;//test
		while (!mat.eof()) {
			mat >> row>>col>>value;
			if (row > 0 && col > 0 && value != 0) {
				element new_element=element(row,col,value);
				matrix[index_matrix]=new_element;
				index_matrix++;
				//cout << row << " " << col << " " << value << endl;
			}
			mat.get();
			if (mat.peek() == EOF) break;
		}
		//cout<<"Process #0 finish reading matrix A!"<<endl;
	}
	
		MPI_Scatterv(matrix,counts,displs,newType,local_matrix,local_n,newType,0,comm);

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
	
	//Barrier 2
	//Now all processes have vector x and a part of A
	//Time to do calculation!
	MPI_Barrier(comm);
	//free useless arrays
	if (my_rank==0){
		free(displs);
		free(counts);
		free(matrix);
	}
	for (int i=0; i<local_n; i++){
		x[local_matrix[i].col] *= local_matrix[i].value;
	}
	//cout<<"Finish multiplication"<<endl;
	
	//Barrier 3
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
		free(displs);
		free(counts);
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
