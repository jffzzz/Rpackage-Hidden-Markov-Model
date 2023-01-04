#include "RcppArmadillo.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <RcppArmadillo.h>
using namespace arma;
//' Evaluation problem - Forward algorithm
//' 
//' @param A initial transition probability
//' @param B initial observation probability
//' @param P initial probability distribution
//' @param O observation sequence
//' @return Observation sequence probability
// [[Rcpp::export]]
double EvaluationForward(arma::mat A,arma::mat B,arma::vec P,arma::vec O){
  int T=O.size();
  int N = P.size();
  mat alpha(T,N);
  double result=0;

  for (int i=0; i<N; i++) {
    alpha(0,i)=P(i)*B(i,O(0));
  }

  for (int t=1; t<T; t++) {
    for (int i=0; i<N; i++) {

      double a=0;
      for (int j=0; j<N; j++)
        a+=alpha(t-1,j)*A(j,i);

      alpha(t,i)=a*B(i,O[t]);
    }
  }

  for (int i=0; i<N; i++)
    result +=alpha(T-1,i);
  return(result);
}
//' Evaluation problem - Backward algorithm
//' 
//' @param A initial transition probability
//' @param B initial observation probability
//' @param P initial probability distribution
//' @param O observation sequence
//' @return Observation sequence probability
// [[Rcpp::export]]
double EvaluationBackward(arma::mat A,arma::mat B,arma::vec P,arma::vec O){
  int N = P.size();
  int T=O.size();
  mat beta(T,N);
  double result=0;

  for (int i=0; i<N; i++) {
    beta(T-1,i)=1;
  }

  for (int t=T-2; t>=0; t--) {
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++)
        beta(t,i) +=A(i,j)*B(j,O[t+1])*beta(t+1,j);
    }
  }

  for (int i=0; i<N; i++)
    result +=P(i)*B(i,O(0))*beta(0,i);
  return(result);
}

// [[Rcpp::export]]
arma::mat LearningForward(arma::mat A,arma::mat B,arma::vec PI,arma::vec O){
  int row=A.n_rows,col=O.n_elem;
  mat alpha(row,col,fill::zeros);
  for(int i=0;i<row;i++){
    alpha(i,0) = PI(i)*B(i,O(0));
  }
  for(int t=1;t<col;t++){
    for(int i=0;i<row;i++){
      double a=0;
      for(int j=0;j<row;j++){
        a+=alpha(j,t-1)*A(j,i);
      }
      alpha(i,t)=a*B(i,O(t));
    }
  }
  return alpha;
}
// [[Rcpp::export]]
arma::mat LearningBackward(arma::mat A,arma::mat B,arma::vec PI,arma::vec O){
  int row=A.n_rows,col=O.n_elem;
  mat beta(row,col,fill::zeros);
  for(int i=0;i<row;i++){
    beta(i,col-1) = 1;
  }
  for(int t=col-2;t>=0;t--){
    for(int i=0;i<row;i++){
      double b=0;
      for(int j=0;j<A.n_cols;j++){
        b+=beta(j,t+1)*A(i,j)*B(j,O(t+1));
      }
      beta(i,t)=b;
    }
  }
  return beta;
}
//' Model parameter learning - Baum-Welch Algorithm - PI
//' 
//' @param A initial transition probability
//' @param B initial observation probability
//' @param PI initial probability distribution
//' @param O observation sequence
//' @param e threshold value
//' @return PI parameter estimation
// [[Rcpp::export]]
arma::vec LearnBaumWelch_PI(arma::mat A,arma::mat B,arma::vec PI,arma::vec O,double e=0.05){
  int row=A.n_rows,col=O.n_elem;
  int done=0;
  while(done==0){
    cube zeta(row,row,col-1);
    mat alpha = LearningForward(A,B,PI,O);
    mat beta = LearningBackward(A,B,PI,O);
    mat gamma(row,col);
    for(int t=0;t<col-1;t++){
      double denominator=0;
      for(int i=0;i<row;i++){
        for(int j=0;j<row;j++){
          denominator+=alpha(i,t)*A(i,j)*B(j,O(t+1))*beta(j,t+1);
        }
      }
      for(int i=0;i<row;i++){
        for(int j=0;j<row;j++){
          zeta(i,j,t)=alpha(i,t)*A(i,j)*B(j,O(t+1))*beta(j,t+1)/denominator;
        }
      }
      for(int i=0;i<row;i++){
        double zetasum=0;
        for(int j=0;j<row;j++){
          zetasum+=zeta(i,j,t);
        }
        gamma(i,t)=zetasum;
      }
    }
    for(int i=0;i<row;i++){
      double final_denominator=0;
      for(int j=0;j<row;j++){
        final_denominator+=alpha(j,col-1)*beta(j,col-1);
      }
      gamma(i,col-1)=alpha(i,col-1)*beta(i,col-1)/final_denominator;
    }
    vec newPI(row);
    for(int i=0;i<row;i++){
      newPI(i)=gamma(i,0);
    }
    mat newA(row,row);
    for(int i=0;i<row;i++){
      for(int j=0;j<row;j++){
        double a1=0,a2=0;
        for(int t=0;t<col-1;t++){
          a1+=zeta(i,j,t);
          a2+=gamma(i,t);
        }
        newA(i,j)=a1/a2;
      }
    }
    mat newB=B;
    mat temp_matrix(1,col,fill::zeros);
    for(int k=0;k<B.n_cols;k++){
      for(int j=0;j<row;j++){
        double b1=0,b2=0;
        for(int t=0;t<col;t++){
          if(O(t)==k){
            temp_matrix(0,t)=1;
          }
          b1+=gamma(j,t)*temp_matrix(0,t);
          b2+=gamma(j,t);
        }
        newB(j,k)=b1/b2;
      }
    }
    done=1;
    for(int i=0;i<newA.n_rows;i++){
      for(int j=0;j<newA.n_cols;j++){
        if(abs(newA(i,j)-A(i,j))>=e){
          done=0;
        }
      }
    }
    for(int i=0;i<newB.n_rows;i++){
      for(int j=0;j<newB.n_cols;j++){
        if(abs(newB(i,j)-B(i,j))>=e){
          done=0;
        }
      }
    }
    for(int i=0;i<newPI.n_elem;i++){
      if(abs(newPI(i)-PI(i))>=e){
        done=0;
      }
    }
    A=newA;
    B=newB;
    PI=newPI;
  }
  return PI;
}
//' Model parameter learning - Baum-Welch Algorithm - A
//' 
//' @param A initial transition probability
//' @param B initial observation probability
//' @param PI initial probability distribution
//' @param O observation sequence
//' @param e threshold value
//' @return A parameter estimation
// [[Rcpp::export]]
arma::mat LearnBaumWelch_A(arma::mat A,arma::mat B,arma::vec PI,arma::vec O,double e=0.05){
  int row=A.n_rows,col=O.n_elem;
  int done=0;
  while(done==0){
    cube zeta(row,row,col-1);
    mat alpha = LearningForward(A,B,PI,O);
    mat beta = LearningBackward(A,B,PI,O);
    mat gamma(row,col);
    for(int t=0;t<col-1;t++){
      double denominator=0;
      for(int i=0;i<row;i++){
        for(int j=0;j<row;j++){
          denominator+=alpha(i,t)*A(i,j)*B(j,O(t+1))*beta(j,t+1);
        }
      }
      for(int i=0;i<row;i++){
        for(int j=0;j<row;j++){
          zeta(i,j,t)=alpha(i,t)*A(i,j)*B(j,O(t+1))*beta(j,t+1)/denominator;
        }
      }
      for(int i=0;i<row;i++){
        double zetasum=0;
        for(int j=0;j<row;j++){
          zetasum+=zeta(i,j,t);
        }
        gamma(i,t)=zetasum;
      }
    }
    for(int i=0;i<row;i++){
      double final_denominator=0;
      for(int j=0;j<row;j++){
        final_denominator+=alpha(j,col-1)*beta(j,col-1);
      }
      gamma(i,col-1)=alpha(i,col-1)*beta(i,col-1)/final_denominator;
    }
    vec newPI(row);
    for(int i=0;i<row;i++){
      newPI(i)=gamma(i,0);
    }
    mat newA(row,row);
    for(int i=0;i<row;i++){
      for(int j=0;j<row;j++){
        double a1=0,a2=0;
        for(int t=0;t<col-1;t++){
          a1+=zeta(i,j,t);
          a2+=gamma(i,t);
        }
        newA(i,j)=a1/a2;
      }
    }
    mat newB=B;
    mat temp_matrix(1,col,fill::zeros);
    for(int k=0;k<B.n_cols;k++){
      for(int j=0;j<row;j++){
        double b1=0,b2=0;
        for(int t=0;t<col;t++){
          if(O(t)==k){
            temp_matrix(0,t)=1;
          }
          b1+=gamma(j,t)*temp_matrix(0,t);
          b2+=gamma(j,t);
        }
        newB(j,k)=b1/b2;
      }
    }
    done=1;
    for(int i=0;i<newA.n_rows;i++){
      for(int j=0;j<newA.n_cols;j++){
        if(abs(newA(i,j)-A(i,j))>=e){
          done=0;
        }
      }
    }
    for(int i=0;i<newB.n_rows;i++){
      for(int j=0;j<newB.n_cols;j++){
        if(abs(newB(i,j)-B(i,j))>=e){
          done=0;
        }
      }
    }
    for(int i=0;i<newPI.n_elem;i++){
      if(abs(newPI(i)-PI(i))>=e){
        done=0;
      }
    }
    A=newA;
    B=newB;
    PI=newPI;
  }
  return A;
}
//' Model parameter learning - Baum-Welch Algorithm - B
//' 
//' @param A initial transition probability
//' @param B initial observation probability
//' @param PI initial probability distribution
//' @param O observation sequence
//' @param e threshold value
//' @return B parameter estimation
// [[Rcpp::export]]
arma::mat LearnBaumWelch_B(arma::mat A,arma::mat B,arma::vec PI,arma::vec O,double e=0.05){
  int row=A.n_rows,col=O.n_elem;
  int done=0;
  while(done==0){
    cube zeta(row,row,col-1);
    mat alpha = LearningForward(A,B,PI,O);
    mat beta = LearningBackward(A,B,PI,O);
    mat gamma(row,col);
    for(int t=0;t<col-1;t++){
      double denominator=0;
      for(int i=0;i<row;i++){
        for(int j=0;j<row;j++){
          denominator+=alpha(i,t)*A(i,j)*B(j,O(t+1))*beta(j,t+1);
        }
      }
      for(int i=0;i<row;i++){
        for(int j=0;j<row;j++){
          zeta(i,j,t)=alpha(i,t)*A(i,j)*B(j,O(t+1))*beta(j,t+1)/denominator;
        }
      }
      for(int i=0;i<row;i++){
        double zetasum=0;
        for(int j=0;j<row;j++){
          zetasum+=zeta(i,j,t);
        }
        gamma(i,t)=zetasum;
      }
    }
    for(int i=0;i<row;i++){
      double final_denominator=0;
      for(int j=0;j<row;j++){
        final_denominator+=alpha(j,col-1)*beta(j,col-1);
      }
      gamma(i,col-1)=alpha(i,col-1)*beta(i,col-1)/final_denominator;
    }
    vec newPI(row);
    for(int i=0;i<row;i++){
      newPI(i)=gamma(i,0);
    }
    mat newA(row,row);
    for(int i=0;i<row;i++){
      for(int j=0;j<row;j++){
        double a1=0,a2=0;
        for(int t=0;t<col-1;t++){
          a1+=zeta(i,j,t);
          a2+=gamma(i,t);
        }
        newA(i,j)=a1/a2;
      }
    }
    mat newB=B;
    mat temp_matrix(1,col,fill::zeros);
    for(int k=0;k<B.n_cols;k++){
      for(int j=0;j<row;j++){
        double b1=0,b2=0;
        for(int t=0;t<col;t++){
          if(O(t)==k){
            temp_matrix(0,t)=1;
          }
          b1+=gamma(j,t)*temp_matrix(0,t);
          b2+=gamma(j,t);
        }
        newB(j,k)=b1/b2;
      }
    }
    done=1;
    for(int i=0;i<newA.n_rows;i++){
      for(int j=0;j<newA.n_cols;j++){
        if(abs(newA(i,j)-A(i,j))>=e){
          done=0;
        }
      }
    }
    for(int i=0;i<newB.n_rows;i++){
      for(int j=0;j<newB.n_cols;j++){
        if(abs(newB(i,j)-B(i,j))>=e){
          done=0;
        }
      }
    }
    for(int i=0;i<newPI.n_elem;i++){
      if(abs(newPI(i)-PI(i))>=e){
        done=0;
      }
    }
    A=newA;
    B=newB;
    PI=newPI;
  }
  return B;
}
//' Prediction problem - Viterbi Algorithm
//' 
//' @param A initial transition probability
//' @param B initial observation probability
//' @param PI initial probability distribution
//' @param V observation set
//' @param Q state set
//' @param O observation sequence
//' @return Optimal path
// [[Rcpp::export]]
arma::mat Viterbi_cpp(arma::mat A,arma::mat B, arma::vec PI,arma::vec V,arma::vec Q,arma::vec obs){
  int N = Q.n_elem;
  int T = obs.n_elem;
  int Z = B.n_cols;
  mat delta(T,N,fill::zeros);
  mat phi(T,N,fill::zeros);
  double P =0.0;

  int Vind0 = 0;
  for(int i=0; i<N; i++){
    for(int j=0; j<Z; j++){
      if(V(j) == obs(0)){
        Vind0 = j;
      }
    }
    delta(0,i) = PI(i) * B(i,Vind0);
    phi(0,i) = 0;
  }

  for(int i=1; i<T; i++){
    for(int j=0; j<N; j++){
      vec tmp(N,fill::zeros);
      for(int k=0; k<N; k++){
        tmp(k) = delta(i-1,k) * A(k,j);
      }
      int Vindi = 0;
      for(int l=0; l<Z; l++){
        if(V(l) == obs(i)){
          Vindi = l;
        }
      }
      double maxtmp = max(tmp);
      delta(i,j) = maxtmp * B(j,Vindi);

      int Vindt = 0;
      for(int m=0; m<N; m++){
        if(tmp(m) == maxtmp){
          Vindt = m;
        }
      }
      phi(i,j) = Vindt;
    }
  }

  vec devec(N,fill::zeros);
  for(int i=0; i<N; i++){
    devec(i) = delta(T-1,i);
  }
  P = max(devec);
  int II = 0;
  for(int i=0; i<N; i++){
    if(devec(i) == P){
      II = i;
    }
  }

  vec pathvec(T,fill::zeros);
  pathvec(0) = II;
  int end = 0;
  for(int i=1; i<T; i++){
    end = pathvec(i-1);
    pathvec(i) = phi(T-i,end);
  }
  vec revpath(T,fill::zeros);
  revpath = reverse(pathvec);
  vec hidden_states(T,fill::zeros);
  int prind = 0;
  for(int i=0; i<T; i++){
    prind = revpath(i);
    hidden_states(i) = Q(prind);
  }
  return hidden_states;
}
