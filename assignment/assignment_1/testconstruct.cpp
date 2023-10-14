#include <iostream>
class MyClass{
    int content;
public:
    explicit MyClass(int a);
    explicit MyClass(double a);
};

MyClass::MyClass(int a){
    content = a;
    std::cout << "int a is triggered" << std::endl;
    
}

MyClass::MyClass(double a){
    content = a;
    std::cout << "double a is triggered" << std::endl;
    
}

int main(){
    MyClass(1.3);
}