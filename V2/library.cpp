#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#include "PMC.h"

/*
 * Définitions du Modèle linéaire
 */
DLLEXPORT double *create_linear_model(int input_dim) {
    double *W;
    W = new double[input_dim + 1];
    if (W == nullptr) {
        exit(1);
    }
    for (int i = 0; i < input_dim + 1; i++) {
        W[i] = ((double) rand() / (RAND_MAX / 2)) - 1.0;
    }
    return W;
}

DLLEXPORT double predict_linear_model_regression(double *W, double *sample_inputs, int arr_size_model) {
    double result = W[0] * 1.0;   //bias
    for (int i = 1; i < arr_size_model+1; i++) {
        result += W[i] * sample_inputs[i-1];
    }
    return result;
}

DLLEXPORT void train_regression_pseudo_inverse_linear_model(double *W, double *flattened_dataset_inputs,
                                                            double *flattened_dataset_expected_outputs, int W_len,
                                                            int flattened_dataset_inputs_len,
                                                            int flattened_dataset_expected_outputs_len) {
    int input_dim = W_len - 1;
    int samples_count = int(floor(double(flattened_dataset_inputs_len) / double(input_dim)));
    MatrixXd xMat = tabToMat(flattened_dataset_inputs, flattened_dataset_inputs_len);
    MatrixXd yMat = tabToMat(flattened_dataset_expected_outputs, flattened_dataset_expected_outputs_len);

    xMat = reshape(xMat, samples_count, input_dim);
    MatrixXd one = ones(samples_count);
    xMat = hstack(one, xMat);
    yMat = reshape(yMat, samples_count, 1);
    MatrixXd Z = (((xMat.transpose() * xMat).inverse()) * xMat.transpose()) * yMat;

    for (int i = 0; i < W_len; i++) {
        W[i] = Z(i, 0);
    }
}

DLLEXPORT double predict_linear_model_classification(double *W, double *sample_inputs, int arr_size_model) {
    if (predict_linear_model_regression(W, sample_inputs, arr_size_model) >= 0) {
        return 1.0;
    } else {
        return -1.0;
    }
}

DLLEXPORT void train_classification_rosenblatt_rule_linear_model(double* W,
                                                                 double* flattened_dataset_inputs,
                                                                 double* flattened_dataset_expected_outputs,
                                                                 int W_len,
                                                                 int flattened_inputs_len,
                                                                 double alpha,
                                                                 int epochs){
    for (int it = 0; it < epochs; it++){
        int k = rand() % W_len;
        double Yk = flattened_dataset_expected_outputs[k];
        double* Xk = flattened_dataset_inputs + k * W_len;
        double gXk = predict_linear_model_classification(W, Xk, W_len);
        W[0] += alpha * (Yk - gXk) * 1.0;
        for(int i=1; i < (W_len+1); i++){
            W[k] += alpha * (Yk - gXk)* Xk[i - 1];
        }
    }

//    int input_dim = W_len - 1;
//    int samples_count = flattened_inputs_len / input_dim;
//    for (int it = 0; it < epochs; it++) {
//        int k = ((int) rand() % (samples_count));
//        int first = k * input_dim;
//        int last = (k+1) * input_dim;
//        double* Xk = separer_tab(flattened_dataset_inputs, first, last);
//        double Yk = flattened_dataset_expected_outputs[k];
//        double gXk = predict_linear_model_classification(W, Xk, W_len);
//        W[0] += alpha * (Yk-gXk) * 1.0;
//        for (int j = 1; j < W_len; j++) {
//            W[j] += alpha * (Yk - gXk) * Xk[j-1];
//        }
//    }
}

/*
 * Définitions du PMC
 */
DLLEXPORT PMC *create_PMC_model(int *npl, int npl_len) {
    PMC *model = new PMC[1];
    for (int l = 0; l < npl_len; l++) {
        model->W.push_back(vector<vector<double>>(0));
        if (l == 0) {
            continue;
        } else {
            for (int i = 0; i < (npl[l - 1] + 1); i++) {
                model->W[l].push_back(vector<double>(npl[l] + 1));
                for (int j = 0; j < npl[l] + 1; j++) {
                    model->W[l][i][j] = ((double) rand() / (RAND_MAX / 2)) - 1.0;
                }
            }
        }

    }

    for (int i = 0; i < npl_len; i++) {
        model->d.push_back(npl[i]);
    }

    for (int l = 0; l < npl_len; l++) {
        model->X.push_back(vector<double>(0));
        for (int j = 0; j < (npl[l] + 1); j++) {
            if (j == 0) {
                model->X[l].push_back(1.0);
            } else {
                model->X[l].push_back(0.0);
            }
        }
    }

    for (int l = 0; l < npl_len; l++) {
        model->deltas.push_back(vector<double>(0));
        for (int j = 0; j < (npl[l] + 1); j++) {
            model->deltas[l].push_back(0.0);
        }
    }

    return model;
}

DLLEXPORT int getLengthX(PMC *model) {
    int dernier_indice = (model->X.size()) - 1;
    return (model->X[dernier_indice].size()) - 1;
}

DLLEXPORT double *predict_PMC_model(PMC *model, double *sample_inputs, bool is_classification) {
    model->forward_pass(sample_inputs, is_classification);
    int dernier_indice = (model->X.size()) - 1;
    int taille = (model->X[dernier_indice].size()) - 1;
    auto newtab = new double[taille];

    for (int i = 0; i < taille; i++) {
        newtab[i] = model->X[dernier_indice][i + 1];
    }

    return newtab;
}

DLLEXPORT void train_stochastic_gradient_backpropagation_PMC_model(PMC *model, double *flattened_dataset_inputs,
                                                                   int flattened_dataset_inputs_len,
                                                                   double *flattened_dataset_expected_outputs,
                                                                   double alpha,
                                                                   int iterations_count, bool is_classification) {
    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_dataset_inputs_len,
                                                     flattened_dataset_expected_outputs, is_classification, alpha,
                                                     iterations_count);
}

DLLEXPORT void destroy_my_model(double *W) {
    delete (W);
}
