#include "CalculMatriciel.h"

// SUPPRIME UNE COLONNE
MatrixXd suppColonne(MatrixXd xMat, int rang) {
    MatrixXd newMat(xMat.rows(), xMat.cols() - 1);
    int j = 0;

    for (int i = 0; i < xMat.cols(); i++) {
        if (i == rang) {
            continue;
        }
        newMat.col(j) << xMat.col(i);
        j++;
    }

    return newMat;
}


// TRANSFORME UN TABLEAU EN MATRICE
MatrixXd tabToMat(double *tab, int tab_len) {
    int y = tab_len;
    MatrixXd xMat(1,y);
    for (int i = 0; i < y; i++) {
        xMat(0,i) = tab[i];
    }
    return xMat;
}

// TRANSFORME UNE MATRICE EN TABLEAU
double *matToTab(MatrixXd XTrain) {
    double *X = new double[XTrain.cols() * XTrain.rows()];
    int cursor = 0;
    for (int i = 0; i < XTrain.rows(); i++) {
        for (int j = 0; j < XTrain.cols(); j++) {
            X[cursor] = XTrain(i, j);
            cursor += 1;
        }
    }
    return X;
}

MatrixXd ones(int rows) {
    MatrixXd result(rows,1);
    for(int i = 0; i < rows; i++) {
        result(i,0) = 1.0;
    }
    return result;
}

MatrixXd reshape(MatrixXd tab, int rows, int col) {
    Map<MatrixXd> temp(tab.data(), col, rows);
    return temp.transpose();
}

MatrixXd hstack(MatrixXd one, MatrixXd X) {
    MatrixXd C(X.rows(), one.cols()+X.cols());
    C << one, X;
    return C;
}


// AJOUTE BIAIS
MatrixXd addBias(MatrixXd xMat) {
    xMat.conservativeResize(xMat.rows(), xMat.cols() + 1);
    for (int i = 0; i < xMat.rows(); i++) {
        xMat.row(i).col(xMat.cols() - 1) << 1;
    }

    return xMat;
}


//SUPPRIME LES LIGNES D'UNE MATRICE DEMANDEES
MatrixXd suppLine(MatrixXd xMat, int listIndex[], int nbIndex) {
    MatrixXd newMat(xMat.rows() - nbIndex, xMat.cols());
    int rows = 0;
    int present;
    for (int i = 0; i < xMat.rows(); i++) {
        present = 0;
        for (int j = 0; j < nbIndex; j++) {
            if (i == listIndex[j]) {
                present = 1;
                break;
            }
        }
        if (present == 0) {
            newMat.row(rows) << xMat.row(i);
            rows += 1;
        }
    }
    return newMat;
}

double* separer_tab(const double* tab, int first, int last) {
    int len;
    len = last - first;
    auto* new_tab = new double[len];
    int i = 0;
    while (i < len) {
        new_tab[i] = tab[first+i];
        i++;
    }
    return new_tab;
}

//FUSIONNE DEUX COLONNES -> UNE COLNNE ENTRE -1 ET 1, POUR CLASSIFICATION
MatrixXd colFus(MatrixXd yMat) {
    MatrixXd newMat(yMat.rows(), 1);
    for (int i = 0; i < yMat.rows(); i++) {
        for (int j = 0; j < yMat.cols(); j += 1) {
            if (yMat(i, j) == 0) {
                newMat(i, j) = -1.0;
            } else {
                newMat(i, j) = yMat(i, j);
            }
        }
    }

    return newMat;
}
