/* fun.c */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h> 

//RUN:
//  gcc -shared -fPIC -o func.so func.c

long double normPdf(double x, double u, double s){
    return (1/(s * sqrt(2 * M_PI))) * exp(-(0.5) * pow((x-u)/s, 2));
}

long double normPdf_stand(double x){
    return (1/(sqrt(2 * M_PI))) * exp(-(0.5) * pow(x, 2.0)) ;
}

long double normCdf_stand(double x, double u, double s){
    return erfc(-(x-u)/(s * sqrt(2)))/2;
}




long double fx1( double x,  double *set, int no_configs) {
    //pdf for untested and tested configs where we need to do the convolution

    //for tested and untested configs in the set 
    //and for only untested configs in the set 

    long double prod_ = 1.0;
    long double sum_ = 0.0;

    long double maxTested = -1.0;

    for(int i=0; i<no_configs; i+=2){
        double u_c = set[i]; //avg value
        double s_c = set[i+1];  //std
        if(s_c == 0.0){ //tested config
            if(u_c > maxTested){ maxTested = u_c;}
        } 
    }

    if(maxTested != -1){

        // there are tested configs
        if(x < maxTested){ 
            return 0;

        }else{
            // only untested configs
            long double prod_normalize = 1.0;
            for(int i=0; i<no_configs; i+=2){
                double s_c = set[i+1];  //std
                if (s_c != 0){
                    double u_c = set[i]; //avg value
                    long double cdf = normCdf_stand(x, u_c, s_c);

                    if (cdf == 0) return 0.0;

                    prod_ *= cdf;
                    prod_normalize *= normCdf_stand(maxTested, u_c, s_c);
                    sum_  += (normPdf_stand((x-u_c)/s_c) / (s_c * cdf));
                }
            }
            long double pdf_untested = sum_ * prod_; //pdf MAX(untested)
            long double cdf_untested = 1.0 - prod_normalize; //cdf MAX(untested)
            //printf("%Lf\n", pdf_untested / cdf_untested );
            if (cdf_untested==0){return 0;}
            return pdf_untested / cdf_untested;
        }
        
    } else{
        //all configs are untested
        for(int i=0; i<no_configs; i+=2){
            double u_c = set[i]; //avg value
            double s_c = set[i+1];  //std


            long double cdf = normCdf_stand(x, u_c, s_c);
            //printf("%lf  %lf  %.10Le \n", u_c, s_c, cdf );

            if (cdf == 0) return 0.0;

            prod_ *= cdf;
            sum_  += (normPdf_stand((x-u_c)/s_c) / (s_c * cdf));
        }
        //printf("%.10Le  %.10Le\n", prod_, sum_ );

    }

    return sum_ * prod_;
}



double f(int n, double *x, void *set) {
    //pdf for untested and tested configs where we need to do the convolution
    // we are solving a double integral

    double *c = (double *) set;

    int size_all = (int) c[0];

    int no_sel_all   = (int) c[1] * 2.0;
    int no_unsel_all = (int) c[2] * 2.0;

    double sel[no_sel_all];
    double unsel[no_unsel_all];

    for(int i=0; i<no_sel_all; i++){
        sel[i] = c[i+3];
    }
    
    for(int i=0; i<no_unsel_all; i++){
        unsel[i] = c[i+no_sel_all+3];
    }

    // for(int i=0; i<no_sel_all; i++) printf("%lf ", sel[i]);
    // printf("\n");
    // for(int i=0; i<no_unsel_all; i++) printf("%lf ", unsel[i]);
    // printf("\n\ndads\n");

    double y = x[1];
    double k = x[0];
   
    //UNSEL = [[0.3, 0.4]]
    //SEL   = [[0.8, 0.2]]
    long double f_unsel = fx1(k, unsel, no_unsel_all);
    long double f_sel   = fx1(k-y, sel, no_sel_all);
    //printf("%f", f_unsel * f_sel );

    //printf("y=%lf  k=%lf  return=%Lf  unsel=%Lf  sel=%Lf;\n" , y, k, y * f_unsel * f_sel, f_unsel, f_sel);

    return y * f_unsel * f_sel;
}



double fd(int n, double *x, void *set) {
    // one of sets is a dirac (ie only has tested configs)

    double *c = (double *) set;

    bool selDirac = false;
    bool unselDirac = false;

    int size_all     = (int) c[0];
    int no_sel_all   = (int) c[1] * 2.0;
    int no_unsel_all = (int) c[2] * 2.0;

    double sel[no_sel_all];
    double unsel[no_unsel_all];

    double y = x[0];
    double x_max = 0;

    //for(int i=0; i<size_all; i++) printf("%lf ", c[i]);
    //printf("\n\n");

    if(no_sel_all == 0){
        //only tested in sel
        selDirac = true;
        no_sel_all = 1;
        x_max = c[3];
    }else{
        for(int i=0; i<no_sel_all; i++){
            sel[i] = c[i+3];
        }
    }

    if(no_unsel_all == 0){
        unselDirac = true;
        no_unsel_all = 1;
        x_max = c[3+no_sel_all];
    }else{
        for(int i=0; i<no_unsel_all; i++){
            unsel[i] = c[i+no_sel_all+3];
        } 
    }
    
    long double f_set = 0;

    if(unselDirac && selDirac){
        return 0;

    }else if(!unselDirac && selDirac){
        //sel is a dirac (only has tested)
        //unsel has untested

        f_set = fx1(y+x_max, unsel, no_unsel_all);

    }else if(unselDirac && !selDirac){
        //unsel is a dirac (only has tested)
        //sel has untested
        //for(int i=0; i<no_sel_all; i++) printf("%lf ", sel[i]);
        //printf("%lf", x_max);
        //printf("\n\n");
        f_set = fx1(-y+x_max, sel, no_sel_all);

    }else{
        printf("APRENDE A PROGRAMAR!!!");
    }

    return y * f_set;
}




/*
long double fx1( double x,  double *set, int no_configs) {

    long double prod_ = 1;
    long double sum_ = 0;

    for(int i=0; i<no_configs; i+=2){
        double u_c = set[i]; //avg value
        double s_c = set[i+1];  //std

        long double cdf = normCdf_stand(x, u_c, s_c);

        if (cdf == 0) return 0;

        prod_ *= cdf;
        sum_  += (normPdf_stand((x-u_c)/s_c) / (s_c * cdf));
    }

    return sum_ * prod_;
}

double f(int n, double *x, void *set) {

    double *c = (double *) set;

    int size_all     = (int) c[0];
    int no_sel_all   = (int) c[1] * 2.0;
    int no_unsel_all = (int) c[2] * 2.0;

    double sel[no_sel_all];
    double unsel[no_unsel_all];


    for(int i=0; i<no_sel_all; i++){
        sel[i] = c[i+3];
    }
    
    for(int i=0; i<no_unsel_all; i++){
        unsel[i] = c[i+no_sel_all+3];
    }

    // for(int i=0; i<no_sel_all; i++) printf("%lf ", sel[i]);
    // printf("\n");
    // for(int i=0; i<no_unsel_all; i++) printf("%lf ", unsel[i]);
    // printf("\n\ndads\n");

    double y = x[1];
    double k = x[0];
 
    long double f_unsel = fx1(k, unsel, no_unsel_all);
    long double f_sel   = fx1(k-y, sel, no_sel_all);

    return y * f_unsel * f_sel;
}
*/
// int main(){
//     double x[] = {0.0, 1.0};
//     double a = f(0.0, x);
//     //printf("%lf\n", a);

// }


