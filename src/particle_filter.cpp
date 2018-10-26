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
    num_particles = 20;
    
    default_random_engine gen;
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
	for (int i = 0; i < num_particles; i++) {
		Particle ptc = { i+1, dist_x(gen), dist_y(gen), dist_theta(gen), 1 };
		particles.push_back(ptc);
        weights.push_back(ptc.weight);
	}
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    
    for (int i = 0; i < num_particles; i++) {
        
        //Predicted new particle
        double x_new;
        double y_new;
        double theta_new;
        
        //Differentiate prediction model equations based on whether yaw_rate is close to 0 or not
        if (fabs(yaw_rate) < 0.0001)
        {
            x_new = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_new = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            theta_new = particles[i].theta;
        }
        else
        {
            x_new = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            y_new = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            theta_new = particles[i].theta + yaw_rate * delta_t;           
        }

        normal_distribution<double> dist_x(x_new, std_pos[0]);
        normal_distribution<double> dist_y(y_new, std_pos[1]);
        normal_distribution<double> dist_theta(theta_new, std_pos[2]);
        
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int size1 = predicted.size();
	int size2 = observations.size();

	for (int i = 0; i < size2; i++) {

		double distance_min = 100000;
		double distance = 0;
		int id_min = 1;

        //Find the predicted landmark which has the minimum distance with each observation, copy the nearest landmark id to observation
		for (int j = 0; j < size1; j++) {

			distance = sqrt( pow(predicted[j].x - observations[i].x, 2) + pow(predicted[j].y - observations[i].y, 2));

			if (distance_min > distance) {
				distance_min = distance;
				id_min = predicted[j].id;
			}

		}

		observations[i].id = id_min;

	}
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

    double weights_sum = 0.0;
    
    for (int i = 0; i < num_particles; i++) {

        vector<LandmarkObs> observations_map;
        double obs_distance;
        //Transformations of landmark observations given in vehicle's coordinate system 
        //into landmark observations in map's coordinate system with predicted particle location in map's coordinate system
        for (int j = 0; j < observations.size(); j++) {
            
            LandmarkObs obs_map;
            
            obs_map.x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
            obs_map.y = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
            observations_map.push_back(obs_map);               

        }
        
        //Limit landmarks from all map landmarks to those only within sensor range of predicted particles
        vector<LandmarkObs> predicted_landmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
            if ((fabs(particles[i].x - landmark.x_f) <= sensor_range) && (fabs(particles[i].y - landmark.y_f) <= sensor_range))
            {
                predicted_landmarks.push_back(LandmarkObs {landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }
        
        //Associate observations to nearest landmarks
        dataAssociation(predicted_landmarks, observations_map);
        
        //Reset the weight of each particle to 1.0 to prepare for new weight update
        particles[i].weight = 1.0;
        
        //Calculate weight for each particle
        for (int j = 0; j < observations_map.size(); j++) {
            
            int obs_id = observations_map[j].id;
            
            for (int k = 0; k < predicted_landmarks.size(); k++)
            {
                int landmark_id = predicted_landmarks[k].id;
                if (obs_id == landmark_id)
                {
                    double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
                    double exponent1 = pow(observations_map[j].x - predicted_landmarks[k].x, 2) / (2 * std_landmark[0] * std_landmark[0]); 
                    double exponent2 = pow(observations_map[j].y - predicted_landmarks[k].y, 2) / (2 * std_landmark[1] * std_landmark[1]);
                    double exponent = exponent1 + exponent2;
            
                    particles[i].weight *= (gauss_norm * exp(-exponent));                  
                }
            }

        }
        
        weights_sum += particles[i].weight;
    }
    
    //Normalize vector weights  
    for (int i = 0; i < particles.size(); i++) {
        particles[i].weight /= weights_sum;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine generator;
	std::discrete_distribution<int> distribution(weights.begin(), weights.end()); 

    vector<int> resample_id;
    for (int i=0; i<num_particles; i++) {
        resample_id.push_back(i);
    }
    
    //Resample particles' id based on particles' weight using discrete_distribution function
	for (int i = 0; i < weights.size(); ++i) {
		resample_id[i] = distribution(generator);
		//cout << resample_id[i] << endl;
	}    

	//for (int i = 0; i < weights.size(); ++i) {
	//	cout << resample_id[i] << endl;
	//}

	for (int i = 0; i < particles.size(); ++i) {
		particles[i] = particles[resample_id[i]];
	}
    
}

Particle SetAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y)
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
