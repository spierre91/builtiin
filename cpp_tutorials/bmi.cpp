#include <iostream>
#include <string>
using namespace std;

float BMI_calculator(float weight, float height){
        if (height == 0){
        throw runtime_error("You attempted to calculate BMI with an invalid value of zero for height \n");
        }
        return weight/(height*height);
}




// Main() function: where the execution of program begins
int main()
{
    string name;
    float weight;
    float height;
    float bmi;

    cout << "Please enter your name \n";
    cin >> name;
    cout << "Hello " << name << ", please enter your weight in Kg\n";
    cin >> weight;
    cout << "Thank you " << name << ", now please enter your height in meters \n";
    cin >> height;

    try{
        bmi = BMI_calculator(weight, height);
        cout << "Your BMI is: " << bmi <<"\n";

    }
    catch (runtime_error& e){
        cout << "Warning: " << e.what();
    }


}
