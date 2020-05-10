//
//  homework4.cpp
//  803
//
//  Created by Zehao Dong on 10/09/19.
//  Copyright © 2019 Zehao Dong. All rights reserved.
//

#include <iostream>
#include <math.h>
class Bond{
public:
    int T;
    float YTM;
    float FV;
    float P;
    float D;
    float Convexity;
    Bond(int T_, float YTM_, float FV_=100){
        T = T_;
        YTM = YTM_;
        FV = FV_;
        P = price(YTM);
        D = duration();
        Convexity = convexity();
    };
    
    float price(float YTM_){
        float p = FV / pow(1 + YTM_, T);
        return p;
    };
    
    float duration(){
        float delta = 0.001;
        float d = (-price(YTM + delta) + price(YTM - delta)) / (2 * delta * price(YTM));
        return d;
    };
    
    float convexity(){
        float delta = 0.001;
        float convexity = (price(YTM + delta) + price(YTM - delta) - 2 * price(YTM)) / (delta * delta * price(YTM));
        return convexity;
    };
};

class Coupon_bond: public Bond{
public:
    float C;
    float P;
    int position;

    Coupon_bond(int T_, float YTM_, float C_=0, float FV_=100, int position_=0)
            :Bond(T_, YTM_, FV_){
        C = C_;
        P = price(YTM);
        D = duration();
        Convexity = convexity();
        position = position_;
    };

    float price(float YTM_){
        float p = 0;
        for(int i=1; i<=T; i++){
            p += (FV * C) / pow((1 + YTM_), i);
        }

        p += FV / pow((1 + YTM_), T);
        return p;
    };

    float duration(){
        float delta = 0.001;
        float d = (-price(YTM + delta) + price(YTM - delta)) / (2 * delta * price(YTM));
        return d;

    };

    float convexity(){
        float convexity = 0;
        float delta = 0.001;
        convexity = (price(YTM + delta) + price(YTM - delta) - 2 * price(YTM)) / (delta * delta * price(YTM));
        return convexity;
    };
};

float port_value(Coupon_bond portfolio[], int size){
    float v=0;
    for (int i=0; i < size; i++){
        v += portfolio[i].price(portfolio[i].YTM) * portfolio[i].position;
    }
    return v;
};

float port_D(Coupon_bond portfolio[], int size){
    float v = port_value(portfolio, size);
    float D = 0;
    for (int i=0; i < size; i++){
        D += portfolio[i].position * portfolio[i].D * portfolio[i].P / v;
    }
    return D;
}

float port_convexity(Coupon_bond portfolio[], int size){
    float v = port_value(portfolio, size);
    float convexity = 0;
    for (int i=0; i < size; i++){
        convexity += portfolio[i].position * portfolio[i].Convexity * portfolio[i].P / v;
    }
    return convexity;
}

int main(){
    Bond test[6]={
        Bond(1, 0.025),
        Bond(2, 0.026),
        Bond(3, 0.027),
        Bond(5, 0.03),
        Bond(10, 0.035),
        Bond(30, 0.04),
    };
//(a)
    std::cout << "(a). Price for zero coupon bond:" << std::endl;
    for(int i=0; i<=5; i++){
        std::cout << test[i].P << std::endl;
    }
//(b)
    std::cout << "(b). duration for zero coupon bond:" << std::endl;
    for(int i=0; i<=5; i++){
        std::cout << test[i].D << std::endl;
    }
//(c)
    Coupon_bond test2[6]={
        Coupon_bond(1, 0.025, 0.03),
        Coupon_bond(2, 0.026, 0.03),
        Coupon_bond(3, 0.027, 0.03),
        Coupon_bond(5, 0.03, 0.03),
        Coupon_bond(10, 0.035, 0.03),
        Coupon_bond(30, 0.04, 0.03),
    };
    std::cout << "(c). Price for 3% coupon bond:" << std::endl;
    for(int i=0; i<=5; i++){
        std::cout << test2[i].P << std::endl;
    }
//(d)
    std::cout << "(d). Duration for 3% coupon bond:" << std::endl;
    for(int i=0; i<=5; i++){
        std::cout << test2[i].D << std::endl;
    }
//(e)
    std::cout << "(e). Convexity for zero coupon bond:" << std::endl;
    for(int i=0; i<=5; i++){
        std::cout << test[i].Convexity << std::endl;
    }
    std::cout << "convexity for 3% coupon bond:" << std::endl;
    for(int i=0; i<=5; i++){
        std::cout << test2[i].Convexity << std::endl;
    }
//(f)
    Coupon_bond portfolio[3]={
        Coupon_bond(1, 0.025, 0, 100, 1),
        Coupon_bond(2, 0.026, 0, 100, -2),
        Coupon_bond(3, 0.027, 0, 100, 1),
    };
    std::cout << "(f). Value of the portfolio:\n" << port_value(portfolio, 3) << std::endl;
//(g)
    std::cout << "(g). Duration of the portfolio:\n" << port_D(portfolio, 3) << std::endl;
    std::cout << "convexity of the portfolio:\n" << port_convexity(portfolio, 3) << std::endl;
//(h)
    Coupon_bond portfolio2[3]={
        Coupon_bond(1, 0.035, 0, 100, 1),
        Coupon_bond(2, 0.036, 0, 100, -2),
        Coupon_bond(3, 0.037, 0, 100, 1),
    };
    std::cout << "(h). Value of the portfolio:\n" << port_value(portfolio2, 3) << std::endl;
//(i)
    Coupon_bond portfolio3[3]={
        Coupon_bond(1, 0.015, 0, 100, 1),
        Coupon_bond(2, 0.016, 0, 100, -2),
        Coupon_bond(3, 0.017, 0, 100, 1),
    };
    std::cout << "(i). Value of the portfolio:\n" << port_value(portfolio3, 3) << std::endl;
//(j)
    std::cout << "(j). Value of the portfolio:" << std::endl;
    float cashflow[5];
    for (int i=0; i<5; i++){
        cashflow[i] = 0.2 * 100 + 0.03 * 100;
        std::cout << "year" << i+1 << ":" << cashflow[i] << std::endl;
    }
//(k)
    float YTM [5] = {0.025, 0.026, 0.027, 0.0285, 0.03};
    float p = 0;
    for(int i=0; i<5; i++){
        p += cashflow[i] / pow((1 + YTM[i]), i + 1);
    }
    std::cout << "(k). Price of the amortizing bond:" << p << std::endl;
    float delta =0.001;
    float p1 = 0;
    for(int i=0; i<5; i++){
        p1 += cashflow[i] / pow((1 + YTM[i] + delta), i + 1);
    }
    float p2 = 0;
    for(int i=0; i<5; i++){
        p2 += cashflow[i] / pow((1 + YTM[i] - delta), i + 1);
    }
    float D = (-p1 + p2) / (2 * delta * p);
    std::cout << "duration of the amortizing bond:" << D << std::endl;
    
    return 0;
};



