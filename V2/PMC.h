#ifndef V1_PMC_H
#define V1_PMC_H

#include <math.h>
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "CalculMatriciel.h"

using namespace std;
using namespace Eigen;

typedef struct PMC_model {
    vector<vector<vector<double>>> W;
    vector<int> d;
    vector<vector<double>> X;
    vector<vector<double>> deltas;

    void forward_pass(double* sample_inputs, bool is_classification) {
        int L = d.size() - 1;

        for (int j = 1; j < (d[0] + 1); j++) {
            X[0][j] = sample_inputs[j-1];
        }

        for (int l = 1; l < (L+1); l++) {
            for (int j = 1; j < d[l] + 1; j++) {
                double sum_result = 0.0;
                for (int i = 0; i < (d[l-1] + 1); i++) {
                    sum_result += W[l][i][j] * X[l-1][i];
                }
                X[l][j] = sum_result;
                if (is_classification || l < L) {
                    X[l][j] = double(tanh(X[l][j]));
                }
            }
        }
    }

    void train_stochastic_gradient_backpropagation(double* flattened_dataset_inputs, int flattened_dataset_inputs_len, double* flattened_dataset_expected_outputs, bool is_classification, double alpha, int iterations_count){
        int input_dim = d[0];
        int lastelement = d.size() - 1;
        int output_dim = d[lastelement];
        int sample_count = int(floor(double(flattened_dataset_inputs_len) / double(input_dim)));
        int L = d.size() - 1;

        for(int it = 0; it < iterations_count; it ++){
            int k = rand()%(sample_count);

            double* sample_input = separer_tab(flattened_dataset_inputs, (k * input_dim), ((k+1) * input_dim));
            double* sample_expected_output = separer_tab(flattened_dataset_expected_outputs, (k * output_dim), ((k+1) * output_dim));
            forward_pass(sample_input, is_classification);

            for(int j = 1; j < d[L] + 1; j++){
                deltas[L][j]=(X[L][j] - sample_expected_output[j-1]);
                if(is_classification) {
                    deltas[L][j] *= (1 - X[L][j] * X[L][j]);
                }
            }

            for(int l = L; l > 0 ; l--){
                for(int i = 1; i < d[l - 1] + 1; i++){
                    double sum_result = 0.0;
                    for(int j = 1; j < d[l] + 1; j++){
                        sum_result += W[l][i][j] * deltas[l][j];
                    }
                    deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * sum_result;
                }
            }

            for(int l = 1; l < L + 1; l++){
                for(int i = 0; i < d[l-1] + 1; i++){
                    for(int j = 1; j < d[l] + 1; j++){
                        W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];
                    }
                }
            }
        }
    }

}PMC;

#endif //V1_PMC_H