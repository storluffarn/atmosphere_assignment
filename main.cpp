
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <armadillo>
#include <cmath>

using namespace std;
typedef unsigned int uint;

void ping(const int line) {cout << "ping at line " << line << endl;}

struct gases
{
	struct gas
	{
		double ratio;
		double sigma;
	};
	
	gas h2o;
	gas co2;
	gas air;
};

double gravity (double r)
{
	double mearth = 5.97219e24;
	double rearth = 6371000;
	double gconstant = 6.67408e-11;
	
	double g = gconstant * mearth / pow(rearth+r,2);

	return g;
}

void makegrid(double rmax, uint steps, vector <double>* grid)
{
	grid->reserve(steps);
	double dr = rmax / steps;
	
	for (uint k = 0; k <= steps; k++)
	{
		double r = k*dr;

		grid->push_back(r);
	}
}


void dens (double temp, uint steps, vector <double>* grid, vector <double>* rhovec)		// assuming A = 1 m2;
{
	rhovec->reserve(steps);

	// constants
	double atm = 101325;
	double R = 8.3144598;
	double molarair = 0.0289644;
	double L = 0.0065;
	double g = 9.82;
	
	for (uint k = 0; k < grid->size()-1 + 1; k++)
	{
		double r = grid->at(grid->size() - 1 - k);

		double T = temp - L*r;
		double c1 = 1.0 - L*r/temp;
		double c2 = g*molarair/(R*L);

		double p = atm * pow(c1,c2);

		double rho = p*molarair/(R*T);

		rhovec->push_back(rho);

		//cout << rho << endl;
	}

}

void visflux (double flux0, double sigma, vector <double>* grid, vector <double>* rhovec, vector <double>* phi)
{
	phi->reserve(rhovec->size());

	//cout << "flux" << endl;
	for (auto& el : (*rhovec))
	{
		auto k = &el - &(*rhovec)[0];

		double r = grid->at(k);
		double rho = rhovec->at(k);

		double t = flux0*exp(-sigma*rho*r);

		//cout << r << " " << rho << " " << t << endl;
	
		phi->push_back(t);
	}
}

void irflux (double flux0, double sigma, vector <double>* grid, vector <double>* rhovec, vector <double>* phi)
{
	phi->reserve(rhovec->size());

	//cout << "flux" << endl;
	for (auto& el : (*rhovec))
	{
		auto k = &el - &(*rhovec)[0];

		double r = grid->at(k);
		double rho = rhovec->at(k);

		double t = flux0*exp(-sigma*rho*r);

	//	cout << r << " " << rho << " " << t << endl;
	
		phi->push_back(t);
	}
}

//void reflection (double flux, uint steps)
//{
//	arma::mat refs(steps,steps,arma::fill::zeros);
//
//	for (uint k = 0; k < steps; k++)
//		for (uint j = 0; j < steps; j++)
//		{
//			refs(k,j) = pow(0.5,abs(k-j + 1));
//		}
//}

void buildmatrix (vector <double>* rhoair, vector <double>* rhoh2o, vector <double>* rhoco2, double sigmaair, double sigmah2o, double sigmaco2, double fluxtopvis, double fluxtopir, vector <double>* phi, double h, uint steps, arma::mat* atmos, arma::vec* rhs)			// with uniform atmosphere composition
{
	arma::rowvec row((steps+1)*4,arma::fill::zeros);

	//flux0vis = phi->at(phi->size()-1);
	auto derp = phi->at(phi->size()-1);
	derp++;

	// build matrix
	for (uint k = 4; k < (steps)*4; k+=4)
	{	
		double a = exp(-sigmaair*(*rhoair)[k/4]*h);
		double b = exp(-sigmah2o*(*rhoh2o)[k/4]*h);
		double c = exp(-sigmaco2*(*rhoco2)[k/4]*h);
		double d = b*c;
		
		//cout << "a: "  << a<< endl;
		//cout << "d: "  << d<< endl;
		
		auto visrow = row;	// visible
		visrow(k) = -1; 
		visrow(k+4) = a;

		auto inrow = row;	// ir in
		inrow(k+1) = -1;
		inrow(k+5) = d;
		inrow(k+7) = 0.5*d;
		
		auto outrow = row;	// ir out
		outrow(k+2)	= -1;
		outrow(k-2) = d;
		outrow(k-1) = 0.5*d;

		auto erow = row;	// emission
		erow(k+3) = -1;
		erow(k-1) = 0.5 * (1-d);
		erow(k+7) = 0.5 * (1-d);
		//erow(k-1) = 0.5 * (1-b)*(1-c);
		//erow(k+7) = 0.5 * (1-b)*(1-c);
		erow(k-2) = 1 - d;
		erow(k+5) = 1 - d;
		erow(k+4) = 1 - a;

		atmos->insert_rows(atmos->n_rows,visrow);
		atmos->insert_rows(atmos->n_rows,inrow);
		atmos->insert_rows(atmos->n_rows,outrow);
		atmos->insert_rows(atmos->n_rows,erow);
	}

	// boundary conditions

	for (uint k = 0; k < 4; k++)
	{
		atmos->insert_rows(0,row);
		atmos->insert_rows(atmos->n_rows,row);
	}
	
	(*rhs) = arma::vec(atmos->n_rows,arma::fill::zeros);

	// bot of matrix, top of atmosphere

	double at = exp(-sigmaair*(*rhoair)[0]*h);
	double bt = exp(-sigmah2o*(*rhoh2o)[0]*h);
	double ct = exp(-sigmaco2*(*rhoco2)[0]*h);
	double dt = bt*ct;
	//cout << "dt: "  << dt << endl;

	// -x-y-z, where: 
	// -x is which element, -4 is vis in etc, 
	// -y is go to cell below
	// -z is go to this element in the cell below
	(*atmos)(atmos->n_rows-4,atmos->n_cols-4-0-0) = 1;	// vis in
	
	(*atmos)(atmos->n_rows-3,atmos->n_cols-3-0-0) = 1;	// ir in
	
	(*atmos)(atmos->n_rows-2,atmos->n_cols-2-0-0) = -1;	// ir out
	(*atmos)(atmos->n_rows-2,atmos->n_cols-2-3-1) = dt;	
	(*atmos)(atmos->n_rows-2,atmos->n_cols-2-3-0) = 0.5*dt;	
	
	(*atmos)(atmos->n_rows-1,atmos->n_cols-1-0-0) = -1;	// emision
	(*atmos)(atmos->n_rows-1,atmos->n_cols-1-4-0) = 0.5*(1-dt);	
	(*atmos)(atmos->n_rows-1,atmos->n_cols-1-4-1) = 1-dt;	
	
	(*rhs)(rhs->n_rows-4) = fluxtopvis*at;
	(*rhs)(rhs->n_rows-3) = fluxtopir*dt;
	(*rhs)(rhs->n_rows-2) = 0;
	(*rhs)(rhs->n_rows-1) = 0;
	
	// top of matrix, bot of atmosphere

	// make space for extra flux equation
	arma::colvec extrarows(atmos->n_rows,arma::fill::zeros);
	atmos->insert_cols(0,extrarows);
	
	arma::rowvec extracols(atmos->n_cols,arma::fill::zeros);
	atmos->insert_rows(0,extracols);
	
	arma::rowvec extrarhs(1,arma::fill::zeros);
	rhs->insert_rows(0,extrarhs);
	
	double ab = exp(-sigmaair*(*rhoair)[steps-1]*h);
	double bb = exp(-sigmah2o*(*rhoh2o)[steps-1]*h);
	double cb = exp(-sigmaco2*(*rhoco2)[steps-1]*h);
	double db = bb*cb;
	
	//cout << "ab: "  << ab << endl;
	//cout << "db: "  << db << endl;
	
	// +x+y+1
	// x element number element
	// y is elements to next cell
	// z is element in next cell
	// 1 is compensation for extra eqn
	
	(*atmos)(0,0) = -1;
	(*atmos)(0,1) = 1;
	(*atmos)(0,2) = 1;
	(*atmos)(0,4) = 0.5;

	(*atmos)(0+1,0+0+0+1) = -1;		// vis in
	(*atmos)(0+1,0+4+0+1) = ab;
	
	(*atmos)(1+1,1+0+0+1) = -1;		// ir in
	(*atmos)(1+1,1+3+1+1) = db;
	(*atmos)(1+1,1+3+3+1) = 0.5*db;

	(*atmos)(2+1,2+1) = -1;
	(*atmos)(2+1,0) = db;		// ir out

	(*atmos)(3+1,3+0+0+1) = -1;		// emmision
	(*atmos)(3+1,0) = 1-db;
	(*atmos)(3+1,3+1+3+1) = 0.5*(1-db);
	(*atmos)(3+1,3+1+1+1) = 1-db;
	(*atmos)(3+1,3+1+0+1) = 1-ab;

	(*rhs)(0,0) = 0;
	(*rhs)(0+1+1) = 0;
	(*rhs)(1+1+1) = 0;
	(*rhs)(2+1+1) = 0;	
	(*rhs)(3+1+1) = 0;
}

void solve(arma::mat* lhs, arma::vec* rhs, arma::vec* sols)
{
	arma::mat l,u;

	lu(l,u,*lhs);
	arma::mat lulhs = u*l;

	(*sols) = arma::solve(lulhs,(*rhs));
	//(*sols) = arma::solve((*lhs),(*rhs));
}

int main ()
{
	double k0 = 273.15;
	double sboltz = 5.67e-8;

	double fluxtopvis = 344;		// sun vis flux
	double fluxtopir = 0;
	//double flux0ir = 227;
	//double sref = 0.04;		// surface reflect
	//double sabs = 0.49;	// surface absorb
	double sigmaair = 3e-5;
	double sigmah2o = 0.75e-1;
	double sigmaco2 = 0.5e-2;
	double fracco2 = 0.00039 ;	// 0.00039
	double frach2o = 0.0025;	// 0.0025

	//gases mygases;

	double a0 = 0.33;		// albedo
	fluxtopvis *= (1 - a0);		

	double temp = 300.0;
	uint steps = 100;			// number of cells	
	double rmax = 30000;	// end of atmosph
	auto h = rmax / steps;	// hight of cell
	
	vector <double> grid;
	vector <double> rhoair;
	vector <double> rhoh2o;
	vector <double> rhoco2;
	vector <double> phi;
	vector <double> temps;

	makegrid(rmax,steps,&grid);
	dens(temp,steps,&grid,&rhoair);

//	for(uint k = 0; k < 250; k++)
//	{
	
//	double fracco2 = 0.00001 + 0.00001*k;

	rhoh2o = rhoair;
	for_each(rhoh2o.begin(),rhoh2o.end(),[frach2o](double &rho){rho *= frach2o;});
	rhoco2 = rhoair;
	for_each(rhoco2.begin(),rhoco2.end(),[fracco2](double &rho){rho *= fracco2;});
	
	visflux(fluxtopvis,sigmaair,&grid,&rhoair,&phi);	

	//for (auto& el : phi)
	//	cout << el << " ";
	//cout << endl;

	double esttemp1 = pow(phi[phi.size()-1] / sboltz, 0.25);

	cout << "first estimate" << endl << "flux: " << phi[phi.size()-1] << " temp: " << esttemp1-k0 << endl;

	arma::mat atmos;
	arma::vec rhs;
	arma::vec sols;

	buildmatrix (&rhoair,&rhoh2o,&rhoco2,sigmaair,sigmah2o,sigmaco2,fluxtopvis,fluxtopir,&phi,h,steps,&atmos,&rhs);			// with uniform atmosphere composition

	solve(&atmos,&rhs,&sols);

	//cout << "atmosphere matrix " << endl;
	//atmos.print();
	//cout << "rhs " << endl;
	//rhs.print();
	//cout << "sols " << endl;
	//sols.print();
	
	double estflux2 = pow(sols[0] / sboltz, 0.25);
	
	cout << "second estimate " << estflux2-k0 << endl;

	temps.push_back(estflux2-k0);
	//}
	
	ofstream fsdens, fsvis, fstemp;
	fsdens.open("densities.dat"); fsvis.open("visflux.dat"); fstemp.open("temps.dat");
	
	for (auto& el : rhoair)
		fsdens << el << endl;
	for (auto& el : phi)
		fsvis << el << endl;
	for (auto& el : temps)
		fstemp << el << endl;


	fsdens.close(); fsvis.close(); fstemp.close();
}
	
