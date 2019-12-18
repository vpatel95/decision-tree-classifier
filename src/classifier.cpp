#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <random>
#include <chrono>

#include "includes/json.hpp"
#include "includes/decision_tree.hpp"

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

/************************************************************************
 * Extract the feature subset as provided in the config.json            *
 *  1. generate the attributes.csv                                      *
 *  2. Create a new data file with selected attributes and the Label    *
 ***********************************************************************/
void extract_features (json config, vector <string> feature_set) {
    int                     i;
    map <string, string>    feature_mapping;
    ofstream                attr_file;
    feature_mapping["Temperature"] = "3";
    feature_mapping["Humidity"] = "4";
    feature_mapping["Light"] = "5";
    feature_mapping["CO2"] = "6";
    feature_mapping["HumidityRatio"] = "7";
    // feature_mapping["Weekend"] = "8";
    // feature_mapping["WorkingHour"] = "9";
    feature_mapping["Occupancy"] = "8";

    int     feature_set_size = feature_set.size();

    string awk_cmd = "awk -F',' 'FNR >= 2 { print ";
    string cmd;

    system("mkdir -p data/extracted");

    string file_name = config["extracted_data"]["attributes"];
    attr_file.open(file_name);
    for (i = 0; i < feature_set_size; i++) {
        if (attr_file) {
            string stmt = feature_set[i] + "," + \
                          (string)(config["feature_info"][feature_set[i]]);
            attr_file << stmt << endl;
        }
        awk_cmd += "$" + feature_mapping[feature_set[i]] + "\",\"";
    }
    attr_file.close();
    awk_cmd += "$" + feature_mapping["Occupancy"] + " }' ";

    cmd = awk_cmd + (string)config["preprocessed_data"]["train"] + " > " + \
          (string)config["extracted_data"]["train"];

    system(cmd.c_str());

    cmd = awk_cmd + (string)config["preprocessed_data"]["test"] + " > " + \
          (string)config["extracted_data"]["test"];

    system(cmd.c_str());

    cmd = awk_cmd + (string)config["preprocessed_data"]["validation"] + " > " + \
          (string)config["extracted_data"]["validation"];

    system(cmd.c_str());

}

/***********************************************************************************
 * Boosting of Dataset                                                               *
 *  1. Initially choose bag_size_percent of the total data.
 *  2. Build a decision tree. And test it on the data.
 *  3. Increase the selection weight of the falsly classified data.
 *  4. In the next iteration of decision_tree choose the top bag_size_percent
 *     of the total data with higher selection weightage.
 ***********************************************************************************/
vector <example> get_boosted_data (map <int, example>   train_data_idx,
                               map <int, int>       weighted_train_data,
                               double               bag_size_percent) {
    int                 i;
    vector <example>    boosted_data;
    int                 bag_size = (train_data_idx.size() * bag_size_percent)/100;

    multimap <int, int> inverted_map;

    for (auto it = weighted_train_data.begin(); it != weighted_train_data.end(); it++) {
        inverted_map.insert(make_pair(it->second, it->first));
    }

    i = 0;
    for (auto it = inverted_map.rbegin(); it != inverted_map.rend(); it++) {
        if (i >= bag_size) {
            break;
        }
        boosted_data.push_back(train_data_idx[it->second]);
        i++;
    }

    return boosted_data;
}

/***********************************************************************************
 * Bagging of Dataset                                                               *
 *  1. Shuffle the dataset using uniformly-distributed integer random number as     *
 *     seed to the shuffle function                                                 *
 *  2. Choose first bag_size_percent examples for the bag                           *
 ***********************************************************************************/
vector <example> get_bagging_data (vector <example> train_data, double bag_size_percent) {
    int     i;
    vector <example>    bag;
    random_device       r;
    seed_seq            seed{r(), r(), r(), r(), r(), r(), r(), r()};
    mt19937             random_engine(seed);
    int                 bag_size = (train_data.size() * bag_size_percent) / 100;

    shuffle(train_data.begin(), train_data.end(), random_engine);

    for (i = 0; i < bag_size; i++) {
        bag.push_back(train_data[i]);
    }

    return bag;
}

/************************************************************************
 * Read the examples from extracted data file into examples class       *
 ***********************************************************************/
vector <example> get_examples( const string &file, const vector <string> &attr_names) {
    int                         i;
    vector <example>            examples_list;
    vector <vector <string> >   data = reader::read_data(file);

    for (i = 0; i < data.size(); i++) {
        string target_value = data[i][data[i].size() - 1];
        data[i].pop_back();
        examples_list.push_back(example(attr_names, data[i], target_value));
    }
    return examples_list;
}

/************************************************************
 * Print outs the classification results from the tests     *
 ***********************************************************/
void classification_report (map <string, double> stats, string data_type) {
    cout << endl;
    cout << data_type << " Dataset Description" << endl;
    cout << "==============================================" << endl;
    cout << "Total Examples : " << setprecision(7) << stats["total"] << endl;
    cout << "\tPositive Examples : " << setprecision(7) << stats["true_positive"] + stats["false_positive"] << endl;
    cout << "\t\tTrue Positive Examples : " << setprecision(7) << stats["true_positive"] << endl;
    cout << "\t\tFalse Positive Examples : " << setprecision(7) << stats["false_positive"] << endl;
    cout << "\tNegative Examples : " << setprecision(7) << stats["true_negative"] + stats["false_negative"] << endl;
    cout << "\t\tTrue Negative Examples : " << setprecision(7) << stats["true_negative"] << endl;
    cout << "\t\tFalse Negative Examples : " << setprecision(7) << stats["false_negative"] << endl;
    cout << endl << endl;
    cout << "Classification Report" << endl;
    cout << "==============================================" << endl;
    cout << "Accuracy : " << setprecision(3) << stats["accuracy"] << endl;
    cout << "Precision : " << setprecision(3) << stats["precision"] << endl;
    cout << "Recall : " << setprecision(3) << stats["recall"] << endl;
    cout << "f1 : " << setprecision(3) << stats["f1"] << endl;
}

void decision_tree_classification (json config) {
    int                     i, j;
    decision_tree           dt;
    set <string>            target_values;
    vector <string>         attr_names;
    map <string, double>    stats;
    vector <string>         features;

    target_values.insert("0");
    target_values.insert("1");

    dt.add_target_values(target_values);

    auto start_train = high_resolution_clock::now();

    int abcd = config["feature_set"][0].size();
    for(i = 0; i < config["feature_set"][0].size(); i++) {
        features.push_back(config["feature_set"][0][i]);
    }

    extract_features(config, features);

    vector < vector <string> > dat = reader::read_data(config["extracted_data"]["attributes"]);
    for (int i = 0; i < dat.size(); i++) {
        attr_names.push_back(dat[i][0]);
        vector <string> temp;
        int flag = 0;
        for (int j = 1; j < dat[i].size(); j++) {
            if (dat[i][j] == "continuous") {
                flag = 1;
                break;
            } else {
                temp.push_back(dat[i][j]);
            }
        }
        dt.add_attr_info(dat[i][0], temp, flag);
    }

    vector <example> train_data = get_examples(config["extracted_data"]["train"], attr_names);
    dt.build(train_data);

    auto stop_train = high_resolution_clock::now();
    auto training_duration = duration_cast<milliseconds>(stop_train - start_train);

    cout << "Decision Tree Training Time : " << training_duration.count() << endl;

    if (config["display_trees"] == true) {
        dt.print();
    }

    vector <example> test_data = get_examples(config["extracted_data"]["test"], attr_names);
    vector <example> validation = get_examples(config["extracted_data"]["validation"], attr_names);

    stats = dt.test(train_data);

    if (config["verbosity"] >= 2) { 
        classification_report(stats, "Training");
    }

    stats.clear();

    auto start_test = high_resolution_clock::now();

    stats = dt.test(test_data);

    auto stop_test = high_resolution_clock::now();
    auto testing_duration = duration_cast<milliseconds>(stop_test - start_test);

    cout << endl << "Decision Tree Test Time : " << testing_duration.count() << endl;

    classification_report(stats, "Test");
    stats.clear();

    stats = dt.test(validation);

    if (config["verbosity"] >= 3) {
        classification_report(stats, "Test 2");
    }

    stats.clear();
}

map <string, double> test_random_forest(vector <example> test_data, vector <decision_tree> dt ) {
    int     i, j;
    int     true_positive = 0;
    int     false_positive = 0;
    int     true_negative = 0;
    int     false_negative = 0;
    int     y_true = 0;
    double  accuracy;
    double  precision;
    double  recall;
    double  f1;
    string  target_class;
    int     total = test_data.size();

    map <string, double> stats;


    for (i = 0; i < test_data.size(); i++) {
        int occupied = 0;
        int unoccupied = 0;
        string  predicted_class;
        target_class = test_data[i].get_target_class();
        for (j = 0; j < dt.size(); j++) {
            string temp_class;
            temp_class = dt[j].classify(test_data[i], dt[j].root);

            (temp_class == "1") ? occupied++ : unoccupied++;
        }

        (occupied > unoccupied) ? predicted_class = "1" : predicted_class = "0";

        if (target_class == "0") {
            if (target_class == predicted_class) {
                true_negative++;
                y_true++;
            } else {
                false_negative++;
            }
        } else {
            if (target_class == predicted_class) {
                true_positive++;
                y_true++;
            } else {
                false_positive++;
            }
        }
    }

    accuracy = (double)(y_true)/(double)(total);
    recall = (double)(true_positive)/(double)(true_positive + false_negative);
    precision = (double)(true_positive)/(double)(true_positive + false_positive);
    f1 = (2 * precision * recall)/(precision + recall);

    stats["accuracy"] = accuracy;
    stats["precision"] = precision;
    stats["recall"] = recall;
    stats["f1"] = f1;

    stats["y_true"] = y_true;
    stats["true_negative"] = true_negative;
    stats["false_negative"] = false_negative;
    stats["true_positive"] = true_positive;
    stats["false_positive"] = false_positive;
    stats["total"] = total;

    return stats;
}

void random_forest_classification (json config) {
    int                     i, j, k;
    int                     flag = 0;
    int                     num_trees = (int)config["num_trees"];
    int                     prev_accuracy = 0;
    vector <decision_tree>  dt;
    set <string>            target_values;
    vector <string>         attr_names;
    vector <example>        train_data;
    vector <example>        test_data;
    vector <example>        validation_data;

    map <string, double>    stats;
    map <int, example>      train_data_idx;
    map <int, int>          weighted_train_data;

    target_values.insert("0");
    target_values.insert("1");

    auto start_train = high_resolution_clock::now();

    for (i = 0; i < num_trees; i++) {
        vector <string> features;
        decision_tree temp_dt;
        temp_dt.add_target_values(target_values);
        attr_names.clear();

        int feature_set_idx = (i % config["feature_set"].size());

        for(j = 0; j < config["feature_set"][feature_set_idx].size(); j++) {
            features.push_back(config["feature_set"][feature_set_idx][j]);
        }

        extract_features(config, features);

        vector < vector <string> > dat = reader::read_data(config["extracted_data"]["attributes"]);
        for (j = 0; j < dat.size(); j++) {
            attr_names.push_back(dat[j][0]);
            vector <string> temp;
            int flag = 0;
            for (k = 1; k < dat[j].size(); k++) {
                if (dat[j][k] == "continuous") {
                    flag = 1;
                    break;
                } else {
                    temp.push_back(dat[j][k]);
                }
            }
            temp_dt.add_attr_info(dat[j][0], temp, flag);
        }

        if (flag == 0) {
            train_data = get_examples(config["extracted_data"]["train"], attr_names);
            for (j = 0; j < train_data.size(); j++) {
                train_data_idx.insert(make_pair(j, train_data[j]));
                weighted_train_data.insert(make_pair(j, 0));
            }

            flag = 1;
        }

        vector <example> training_data;
        if (config["boosting"]) {
            training_data = get_boosted_data(train_data_idx, weighted_train_data,
                                                (double)config["bag_size_percent"]);
        } else {
            training_data = get_bagging_data(train_data, (double)config["bag_size_percent"]);
        }

        temp_dt.build(training_data);

        if (config["boosting"]) {
            temp_dt.test_rf(training_data, train_data_idx, weighted_train_data);
        }

        if (config["display_trees"] == true) {
            temp_dt.print();
            cout << endl << endl;
        }

        dt.push_back(temp_dt);
    }

    auto stop_train = high_resolution_clock::now();
    auto training_duration = duration_cast<milliseconds>(stop_train - start_train);

    cout << "Random Forest Training Time : " << training_duration.count() << endl;


    test_data = get_examples(config["extracted_data"]["test"], attr_names);
    validation_data = get_examples(config["extracted_data"]["validation"], attr_names);



    stats = test_random_forest(train_data, dt);

    if (config["verbosity"] >= 2) {
        classification_report(stats, "Training");
    }

    stats.clear();

    auto start_test = high_resolution_clock::now();

    stats = test_random_forest(test_data, dt);

    auto stop_test = high_resolution_clock::now();
    auto testing_duration = duration_cast<milliseconds>(stop_test - start_test);

    cout << "Random Forest Test Time : " << testing_duration.count() << endl;

    classification_report(stats, "Test");
    stats.clear();

    stats = test_random_forest(validation_data, dt);

    if (config["verbosity"] >= 3) {
        classification_report(stats, "Test 2");
    }

    stats.clear();
}

int main( int argc, char *argv[]) {

    if( argc != 2 ) {
        cout << "Usage : " << argv[0] << " <config_file>" << endl;
        exit(1);
    }

    int         i;
    json        config;
    ifstream    config_file(argv[1]);

    config_file >> config;

    // extract_features(config);

    if (config["classification_model"] == "decision_tree") {
        decision_tree_classification(config);
    }

    if (config["classification_model"] == "random_forest") {
        random_forest_classification(config);
    }

    return 0;
}
