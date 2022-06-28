#include <iostream>
#include <stdexcept>      // std::out_of_range
#include <vector>
using namespace std;

int main(){
    int months;
    int current_month = 1;
    cout << "Please enter the number of months \n";
    cin >> months;
    try{
       std::vector<float> contributions(months); //float contributions[months];
       float initial_contrbution = 100;
       float sum_contributions = 0;
       contributions[1] = initial_contrbution;
       while (current_month <= months){
        contributions[current_month + 1] =1.02*contributions[current_month];
        cout << "Month " << current_month << " contribution is: " << contributions[current_month]<< endl;
        sum_contributions += contributions[current_month];
        current_month++;
    }
       cout<<"Sum of contributions for " << months << " months is: "<<sum_contributions << endl;
    }
    catch (const std::length_error& le) {
          std::cerr << "Length of " << le.what() << " can't be negative \n";
    }

}
