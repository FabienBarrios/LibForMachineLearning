#ifndef V1_CALCULMATRICIEL_H
#define V1_CALCULMATRICIEL_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;


//int findColinear(MatrixXd xMat);
//MatrixXd suppColonne(MatrixXd xMat, int rang);
MatrixXd tabToMat(double *XTrain, int sampleCount);

double *matToTab(MatrixXd XTrain);

MatrixXd ones(int rows);

MatrixXd reshape(MatrixXd tab, int rows, int col);

MatrixXd hstack(MatrixXd one, MatrixXd X);

double* separer_tab(const double* tab, int first, int last);

MatrixXd addBias(MatrixXd xMat);

MatrixXd suppLine(MatrixXd xMat, int listIndex[], int nbIndex);

MatrixXd colFus(MatrixXd yMat);


#endif