#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <chrono>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Sets the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Adds random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).


    num_particles = 20;

    // construct a random generator engine from a time-based seed:
    unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine generator(seed);

    // parameterize the normal distributions for x, y and theta
    // the mean is the reported x, y GPS coordinates and the heading (yaw) that the simulator sends to the filter
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    // initialize the particles
    for (int i = 0; i < num_particles; i++) {
        Particle p{.id = i, .x = dist_x(generator), .y = dist_y(generator), .theta = dist_theta(
                generator), .weight = 1.};
        particles.push_back(p);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Predicts new particle position and yaw using a simple motion model.
    // Adds random Gaussian noise to particle predicted state

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    for (int i = 0; i < num_particles; i++) {

        if (fabs(yaw_rate) > 0.001) {
            particles[i].x +=
                    (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y +=
                    (velocity / yaw_rate) * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
        } else {
            particles[i].x += velocity * cos(particles[i].theta) * delta_t;
            particles[i].y += velocity * sin(particles[i].theta) * delta_t;
        }

        particles[i].theta = yaw_rate * delta_t;

        // add motion/control model uncertainty
        std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

        particles[i].x += dist_x(generator);
        particles[i].y += dist_y(generator);
        particles[i].theta += dist_theta(generator);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    // NOTE 2: the first parameter has a rather misleading name: it is just the list of landmarks as provided by the map,
    // that you read from file data/map_data.txt;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution.
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    auto num_observations = static_cast<unsigned int>(observations.size());
    auto num_landmarks = static_cast<unsigned int>(map_landmarks.landmark_list.size());

    std::vector<LandmarkObs> observations_gcs;

    for (int i = 0; i < num_particles; i++) {
        cout << "Particle " << i << endl;

        vector<int> associations(num_observations);
        double sum_measurement_likelihoods(0.0);
        for (int m = 0; m < num_observations; m++) {

            // The observations are given in the VEHICLE'S coordinate system. Transform the observation in the global
            // coordinate system. This transformation requires both rotation AND translation (but no scaling).
            //   The following is a good resource for the theory:
            //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
            //   and the following is a good resource for the actual equation to implement (look at equation
            //   3.33 http://planning.cs.uiuc.edu/node99.html

            double observations_gcs_x =
                    observations[m].x * cos(particles[i].theta) - observations[m].y * sin(particles[i].theta);
            double observations_gcs_y =
                    observations[m].x * sin(particles[i].theta) + observations[m].y * cos(particles[i].theta);

            observations_gcs_x += particles[i].x;
            observations_gcs_y += particles[i].y;

            LandmarkObs observation_gcs = {m, observations_gcs_x, observations_gcs_y};

            observations_gcs.push_back(observation_gcs);

            double min_distance = sensor_range;

            for (int l = 0; l < num_landmarks; l++) {

                // associate each observation to a landmark
                double distance_observation_landmark =
                        dist(observations_gcs[m].x, observations_gcs[m].y, map_landmarks.landmark_list[l].x_f,
                             map_landmarks.landmark_list[l].y_f);

                if (distance_observation_landmark < min_distance) {

                    min_distance = distance_observation_landmark;
                    associations[m] = static_cast<unsigned int>(l);
                }
            }

            // Likelihood of each measurement

            /* The mean of the Multivariate-Gaussian is the measurement's associated landmark position and the
             * Multivariate-Gaussian's standard deviation is described by our initial uncertainty in the x and y ranges.
             * The Multivariate-Gaussian is evaluated at the point of the transformed measurement's position.
             * x and y are the observations in map coordinates and μx, μ​y are the coordinates of
             * the nearest landmarks.
             */

            double gauss_norm = 1./(2. * M_PI * std_landmark[0] * std_landmark[1]);

            double exponent= pow(observations_gcs[m].x - map_landmarks.landmark_list[associations[m]].x_f, 2)/(2. * pow(std_landmark[0], 2))
                             + pow(observations_gcs[m].y - map_landmarks.landmark_list[associations[m]].y_f, 2)/(2. * pow(std_landmark[1], 2));

            double measurement_likelihood = gauss_norm * exp(-exponent);

            cout << "Likelihood of measurement " << m << " = " << measurement_likelihood << endl;

            sum_measurement_likelihoods += measurement_likelihood;
            particles[i].weight *= measurement_likelihood;

        }
        // Store the asociations of each particle to the in-range landmarks
        particles[i].associations = associations;

        // Normalize particle weight
        particles[i].weight /= sum_measurement_likelihoods;
        cout << "Weight " <<  " = " << particles[i].weight << endl;
    }

}

template<typename Iter_T>
long double vectorNorm(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0L));
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

//    bool resampling_flag;
//
//    // norm squared of latest particle weights
//    long double newWeightsL2NormSq = pow(vectorNorm(weights.begin(), weights.end()), 2.0);

    // Resampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());

    std::map<int, int> m;
    for(int i=0; i < num_particles; i++) {
        ++m[d(gen)];
    }

    int i=0;
    for(auto p : m) {
        std::cout << "particle " << p.first << " resampled " << p.second << " times\n";
        particles[i++] = particles[p.first];
    }

    for(int i=0; i < num_particles; i++) {
        particles[i] = particles[d(gen)];
    }



}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
