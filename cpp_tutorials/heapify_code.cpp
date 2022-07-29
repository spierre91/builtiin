#include <iostream>

using namespace std;


void heapify(int array_in[], int array_size, int subtree_root_index)
{
    int largest_value = subtree_root_index;
    int left = 2*subtree_root_index + 1;
    int right = 2*subtree_root_index + 2;


    if (left < array_size && array_in[left] > array_in[largest_value]){
    largest_value = left;
}

    if (right < array_size && array_in[right] > array_in[largest_value]){
    largest_value = right;
}


    if (largest_value != subtree_root_index )
{
    swap(array_in[subtree_root_index], array_in[largest_value]);

    heapify(array_in, array_size, largest_value);
}


}


void construct_heap(int array_in[], int array_size){
int last_non_leaf_node = (array_size/2) -1;

for (int subtree_root_index = last_non_leaf_node; subtree_root_index >=0; subtree_root_index -=1)
{
    heapify(array_in, array_size, subtree_root_index);
}

}

void print_heap(int array_in[], int array_size){
   cout << "Printing values at each node in heap" << endl;

   for (int index = 0; index < array_size; index+=1){
       cout<< array_in[index] << endl;

}

}


int main(){
    int array_in[] = { 3, 5, 8, 10, 17, 11, 13, 19, 22, 24, 29};

    int array_size = sizeof(array_in) / sizeof(array_in[0]);

    construct_heap(array_in, array_size);

    print_heap(array_in, array_size);

}
