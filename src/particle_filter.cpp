
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


    num_particles = 10;

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
        weights.push_back(1.);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Predicts  resampled particles new position and yaw using a simple motion model.
    // Adds random Gaussian noise to abstract noisy motion model

    unsigned seed = 100;//std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // add motion/control model uncertainty
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (auto &p : particles) {

        cout << "Previous Position of Particle " << p.id << ": (" << p.x << "," << p.y << ")"
             << endl;

        if (fabs(yaw_rate) > 0.001) {
            p.x +=
                    (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y +=
                    (velocity / yaw_rate) * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta));
        } else {
            p.x += velocity * cos(p.theta) * delta_t;
            p.y += velocity * sin(p.theta) * delta_t;
        }

        p.theta += yaw_rate * delta_t;

        p.x += dist_x(generator);
        p.y += dist_y(generator);
        p.theta += dist_theta(generator);

        cout << "Predicted Position of Particle " << p.id << ": (" << p.x << "," << p.y << ")"
             << endl;
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
    auto num_map_landmarks = static_cast<unsigned int>(map_landmarks.landmark_list.size());


    Map landmarks_in_range;
    landmarks_in_range.landmark_list.clear();
    weights.clear();

    for (auto &p : particles) {
        p.weight = 1.;

        cout << "Particle " << p.id << " performs " << num_observations << " LIDAR measurements" << endl;

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            double distance = dist(p.x, p.y, map_landmarks.landmark_list[j].x_f,
                                   map_landmarks.landmark_list[j].y_f);

            if (distance < sensor_range) {
                landmarks_in_range.landmark_list.push_back(map_landmarks.landmark_list[j]);
            }
        }
        int num_in_range_landmarks = (int) landmarks_in_range.landmark_list.size();

        vector<int> associations(num_observations);
        std::vector<LandmarkObs> observations_gcs;
        for (auto &m : observations) {

            // The observations (lidar measurements) are given in the VEHICLE'S coordinate system. Transform the observation
            // in the global coordinate system (GCS). This transformation requires both rotation AND translation (but no scaling).
            //   The following is a good resource for the theory:
            //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
            //   and the following is a good resource for the actual equation to implement (look at equation
            //   3.33 http://planning.cs.uiuc.edu/node99.html

            double observation_gcs_x =
                    p.x + m.x * cos(p.theta) - m.y * sin(p.theta);
            double observation_gcs_y =
                    p.y + m.x * sin(p.theta) + m.y * cos(p.theta);

            // Store the transformed coordinates
            LandmarkObs observation_gcs = {m.id, observation_gcs_x, observation_gcs_y};
            observations_gcs.push_back(observation_gcs);

            // associate each observation to a landmark within LIDAR sensor range
            double min_distance = sensor_range;
            int association = 999;
            for (int l = 0; l < num_in_range_landmarks; l++) {

                double distance_observation_landmark =
                        dist(observation_gcs_x, observation_gcs_y, landmarks_in_range.landmark_list[l].x_f,
                             landmarks_in_range.landmark_list[l].y_f);

                if (distance_observation_landmark < min_distance) {

                    min_distance = distance_observation_landmark;
                    association = landmarks_in_range.landmark_list[l].id_i;
                    cout << "association = " << association << endl;
                }
            }

            cout << "Measurement "
                 << observation_gcs.id <<
                 " with GCS coordinates (" <<
                 observation_gcs.x << "," <<
                 observation_gcs.y << ")  is associated with landmark ID "
                 << association << " with coordinates (" << map_landmarks.landmark_list[association-1].x_f <<
                 "," << map_landmarks.landmark_list[association-1].y_f << ")" << endl;

            // Likelihood of each measurement

            /* The mean of the Multivariate-Gaussian is the measurement's associated landmark position and the
             * Multivariate-Gaussian's standard deviation is described by our initial uncertainty in the x and y ranges.
             * The Multivariate-Gaussian is evaluated at the point of the transformed measurement's position.
             * x and y are the observations in map coordinates and μx, μ​y are the coordinates of
             * the nearest landmarks.
             */

            double gauss_norm = 1./(2. * M_PI * std_landmark[0] * std_landmark[1]);

            double exponent = pow(observation_gcs_x - map_landmarks.landmark_list[association-1].x_f, 2)/(2. * pow(std_landmark[0], 2))
                             + pow(observation_gcs_y - map_landmarks.landmark_list[association-1].y_f, 2)/(2. * pow(std_landmark[1], 2));

            double measurement_likelihood = gauss_norm * exp(-exponent);

            cout << "Likelihood of measurement " << m.id << " = " << measurement_likelihood << endl;

            p.weight *= measurement_likelihood;

        }

        // Store particle weights
        weights.push_back(p.weight);
        landmarks_in_range.landmark_list.clear();
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight w.
    // We use the std::discrete_distribution where probability of each of the n possible numbers to be produced
    // is their corresponding weight divided by the total of all weights. There is no need for explicit normalization.

    // Resampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());


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
