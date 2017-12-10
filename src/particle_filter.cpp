/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Initializes particles
	// The code is similar to the Section 5 of lesson 14
	// except this time the particles are being added to the vector

	// Set the number of particles.
	num_particles = 100;

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;
		
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle particle;
		particle.id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.0;

		// add a new particle to the set
		particles.push_back(particle);

		weights.push_back(particles[i].weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	/*
	* Section 8, lesson 14
	* The equations for updating x, y and the yaw angle when the yaw rate is not equal to zero:
	* xf = x0 + (v/.Q)[sin(Q0 + .Q(dt)) - sin(Q0)]
	* yf = y0 + (v/.Q)[cos(Q0) - cos(Q0 + .Q(dt))]
	* Qf = Q0 + .Q(dt)
	*/

	default_random_engine gen;

	// random Gaussian noise
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto& particle: particles){
		double dot_theta_delta = yaw_rate*delta_t;

		// avoid division by zero
		// reuse of the prediction step code 
		// from the Section 21 of UKF lesson
		if (fabs(yaw_rate) > 0.001) {
			particle.x = particle.x + (velocity/yaw_rate) * ( sin(particle.theta + dot_theta_delta) - sin(particle.theta));
			particle.y = particle.y + (velocity/yaw_rate) * ( cos(particle.theta) - cos(particle.theta + dot_theta_delta));
		} else {
			particle.x = particle.x + velocity*delta_t*cos(particle.theta);
			particle.y = particle.y + velocity*delta_t*sin(particle.theta);
		}
		particle.theta = particle.theta + dot_theta_delta;

		// add random Gaussian noise
		particle.x = particle.x + dist_x(gen);
		particle.y = particle.y + dist_y(gen);
		particle.theta = particle.theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	int i = 0;
	for (auto& particle: particles) {
		
		// The result of multi-variate Gaussian calculation
		// is a product, so initialize with 1
		long double weight = 1.0;

		for (auto&& observation: observations) {
			
			// Observations in the car coordinate system are transformed 
			// into map coordinates with a homogenous transformation matrix
			double observation_map_x = particle.x + observation.x * cos(particle.theta) 
				- observation.y * sin(particle.theta);
			double observation_map_y = particle.y + observation.x * sin(particle.theta) 
				+ observation.y * cos(particle.theta);

			// find the nearest landmark to the observation
			// using nearest neightbor algorithm
			Map::single_landmark_s min_landmark;
			double min_distance = sensor_range; // distance cannot be more than sensor range

			for (auto& landmark: map_landmarks.landmark_list){

				double x_diff = observation_map_x - landmark.x_f;
				double y_diff = observation_map_y - landmark.y_f;
				double distance = sqrt(x_diff*x_diff + y_diff*y_diff);

				if(distance < min_distance){
					min_landmark = landmark;
					min_distance = distance;
				}
			}

			// Calculate the multi-variate Gaussian
			double x_diff = observation_map_x - min_landmark.x_f;
			double y_diff = observation_map_y - min_landmark.y_f;

			double g = ( exp( -0.5 * (pow(x_diff, 2)/pow(sigma_x, 2) + pow(x_diff,2)/pow(sigma_y, 2)) ) ) / (2*M_PI*sigma_x*sigma_y);

			weight = weight * g;
		}

		particle.weight = weight;
		weights[i] = weight;
		i++;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// the code below is C++ implementation of
	// the resampling wheel from section 20 of the lesson 13

	default_random_engine gen;
	// python's random.random() alternative
	uniform_real_distribution<double> random_double_gen(0.0, 1.0);

	vector<Particle> resampled_particles;
	int index = (int) (random_double_gen(gen) * num_particles);
	double beta = 0.0;
	// python's max() alternative
	double mw = *max_element(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		beta += random_double_gen(gen) * 2.0 * mw;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
