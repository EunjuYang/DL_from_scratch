/*
 *  Simple Perceptron example
 *  input is given as x vector
 *  perceptron has weight vector
 */ 
#include <iostream>
#define DATA_NUMS 4
#define WEIGHT_NUMS 3

// dot product between two float array
float dot(float *v1, float *v2, int len){
    float sum = 0;
    for(int i = 0; i < len; i++){
        sum += v1[i] * v2[i];
    }

    return sum;
}

// step function
float step(float v){
    return v > 0 ? 1 : 0;
}

float forward(float *x, float *w, int len){
    float u = dot(x, w, len);
    return step(u);
}

void train(float *w, float *x, float t, float e, int len){

    // feed forward value
    float z = forward(x, w, len);

    // t is given label & e is learning rate (etta)
    for (int j = 0; j < len; j++){
        w[j] += (t - z) * x[j] * e;
    }
}

int main(){

    // define learning rate (etta)
    float e = 0.1;

    // input data
    float x[DATA_NUMS][WEIGHT_NUMS] = {{1,0,0},{1,0,1},{1,0,1},{1,1,1}};

    // logical product
    float t[DATA_NUMS] = {0, 0, 0, 1};

    // logical sum
    // float t[DATA_NUMS] = {0, 1, 1, 1};

    // initialize weight parameter to zeros
    float w[WEIGHT_NUMS] = {0, 0, 0};

    int epoch = 10;
    for(int i = 0; i < epoch; i++){
        std::cout << "epoch :" << i << " ";
        for(int j = 0; j < DATA_NUMS;  j++){
            train(w, x[j], t[j], e, WEIGHT_NUMS);
        }

        for(int j = 0; j < WEIGHT_NUMS; j++){
            std::cout << "w" << j << ":" << w[j] << " ";
        }
        std::cout << std::endl;
    }

    for(int i = 0; i < DATA_NUMS; i ++){
        std::cout << forward(x[i], w, WEIGHT_NUMS) << " ";
    }
    std::cout << std::endl;

    return 0;


}