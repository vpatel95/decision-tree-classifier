#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <sstream>

#include "includes/decision_tree.hpp"

#define NUM_TARGET_VALUES 2

vector < vector <string>> reader::read_data (string file) {
    ifstream fin(file, ios::in);
    vector < vector <string> > data;
    string s;
    while(fin >> s) {
        vector <string> attr;
        stringstream str(s);
        string temp;
        while (getline(str, temp, ',')) {
            attr.push_back(temp);
        }
        data.push_back(attr);
    }
    return data;
}

set <string> reader::read_target_value (string file, int n) {
    ifstream fin(file, ios::in);
    set <string> attr;
    while (!fin.eof()) {
        string s;
        fin >> s;
        stringstream str(s);
        for (int i = 0; i < n; i++) {
            string temp;
            getline( str, temp, ',');
            if(i == (n-1)) {
                attr.insert(temp);
            }
        }
    }
    return attr;
}

instance::instance() {
    ;
}

instance::instance( const vector <string> &attr_names,
                    const vector <string> &attr_values ) {
    int i;
    for (i = 0; i < attr_names.size(); i++ ) {
        examples_list[attr_names[i]] = attr_values[i];
     }
}

string instance::operator[]( const string &attr_name ) {
   return examples_list[attr_name];
}

example::example() {
    ;
}

example::example( const vector <string> &attr_names,
                  const vector <string> &attr_values,
                  const string          &target_class ) {
    int i;
    for (i = 0; i < attr_names.size(); i++) {
        examples_list[attr_names[i]] = attr_values[i];
    }
    this->target_class = target_class;
}

// bool example::operator==(example ex) {
//     return (this == ex);
// }

string example::get_target_class() const {
    return target_class;
}

decision_tree_node::decision_tree_node() {
    ;
}

void decision_tree_node::set_attribute_name( const string &attr_name ) {
    this->attr_name = attr_name;
}

void decision_tree_node::set_type( const string &type) {
    this->type = type;
}

void decision_tree_node::set_divider( const double &divider ) {
    this->divider = divider;
}

void decision_tree_node::add_value( const string &value ) {
    this->values.push_back(value);
}

string decision_tree_node::get_attribute_name() const {
    return attr_name;
}

string decision_tree_node::get_type() const {
    return type;
}

double decision_tree_node::get_divider() const {
    return divider;
}

vector <string> decision_tree_node::get_values() const {
    return values;
}

decision_tree_node *& decision_tree_node::operator[]( const string &attr_value ) {
    return children[attr_value];
}

vector <decision_tree_node *> decision_tree_node::get_children_pointers() {
    vector <decision_tree_node *> result;
    for (auto it = children.begin(); it != children.end(); it++) {
        result.push_back(it->second);
    }
    return result;
}

void decision_tree::add_attr_info( const string             &attr_name,
                                   const vector <string>    &attr_values,
                                   int                      flag) {
    possible_values[attr_name] = attr_values;
    if (flag == 0)
        attribute_type[attr_name] = "discrete";
    else
        attribute_type[attr_name] = "continuous";
}

void decision_tree::add_target_values( set <string> target_values ) {
    this->target_values = target_values;
}

double decision_tree::calculate_entropy( const map <string, int> &entropy_map) {
    int     i;
    double  sum = 0.0;
    double  ans = 0.0;
    vector <double> temp;
    for (auto it = entropy_map.begin(); it != entropy_map.end(); it++) {
        sum += it->second;
        temp.push_back((double)it->second);
    }

    for (i = 0; i < temp.size(); i++) {
        ans += (temp[i]/sum)*(log2(temp[i]/sum))*(-1.00);
    }

    return ans;
}

double decision_tree::discrete_info_gain( vector <example>  &examples_list,
                                          const string      &attr_name ) {
    int                 i, j;
    set <string>        target_values_set = target_values;
    map <string, int>   attribute_num_target_mapping[NUM_TARGET_VALUES];

    double sum = examples_list.size();

    map <string, int> entropy_map;
    for (i = 0; i < examples_list.size(); i++) {
        entropy_map[examples_list[i].get_target_class()]++;
    }

    double dataset_entropy = calculate_entropy(entropy_map);

    double attribute_entropy = 0.0;
    for (i = 0; i < possible_values[attr_name].size(); i++) {
        double numerator = 0;
        string attribute = possible_values[attr_name][i];
        entropy_map.clear();

        for (j = 0; j < examples_list.size(); j++) {
            if (attribute == examples_list[j][attr_name]) {
                entropy_map[examples_list[j].get_target_class()]++;
                numerator++;
            }
        }

        double temp_entropy = calculate_entropy(entropy_map);
        attribute_entropy += (numerator/sum) * temp_entropy;
    }

    return (dataset_entropy - attribute_entropy);
}

pair <double, double>  decision_tree::continuous_info_gain( vector <example>  &examples_list,
                                            const string      &attr_name ) {
    int                             i, j;
    double                          divider;
    double                          info_gain = -1;
    double                          sum = examples_list.size();
    double                          dataset_entropy;
    double                          attribute_entropy;
    map <string, int>               entropy_map;
    set < pair <double, string> >   continuous_value_set;
    vector < pair <double, string> > continuous_value_list;

    for (i = 0; i < examples_list.size(); i++) {
        continuous_value_set.insert(make_pair(atof(examples_list[i][attr_name].c_str()),
                                              examples_list[i].get_target_class()));
    }

    for (auto it = continuous_value_set.begin(); it != continuous_value_set.end(); it++) {
        continuous_value_list.push_back(*it);
    }

    for (i = 0; i < examples_list.size(); i++) {
        entropy_map[examples_list[i].get_target_class()]++;
    }

    dataset_entropy = calculate_entropy(entropy_map);

    for (i = 1; i < continuous_value_list.size(); i++) {
        if( continuous_value_list[i].second == continuous_value_list[i-1].second) {
            continue;
        } else {
            double numerator = 0;
            entropy_map.clear();
            for (j = 0; j < i; j++) {
                entropy_map[continuous_value_list[j].second]++;
                numerator++;
            }

            attribute_entropy = (numerator/sum) * calculate_entropy(entropy_map);
            entropy_map.clear();
            numerator = 0;

            for (j = i; j < continuous_value_list.size(); j++) {
                entropy_map[continuous_value_list[j].second]++;
                numerator++;
            }

            attribute_entropy += (numerator/sum) * calculate_entropy(entropy_map);

            if ((dataset_entropy - attribute_entropy) > info_gain) {
                info_gain = dataset_entropy - attribute_entropy;
                divider = (continuous_value_list[i].first + continuous_value_list[i-1].first)/2;
            }
        }
    }

    return make_pair(info_gain, divider);
}

void decision_tree::build( const vector <example> &train_data ) {
    vector <string> all_attributes;
    for (auto it = possible_values.begin(); it != possible_values.end(); it++) {
        all_attributes.push_back(it->first);
    }
    build( train_data, root, all_attributes, "0");
}

#define create_leaf_node(__node, majority) {    \
    __node = new decision_tree_node;            \
    __node->set_attribute_name(majority);       \
    __node->set_type("leaf");                   \
    __node->set_divider(0);                     \
}

void decision_tree::build( vector <example>     train_data,
                           decision_tree_node   *&p,
                           vector <string>      check_attr,
                           string               majority ) {

    if (train_data.size() == 0) {
        create_leaf_node(p, majority);
        return;
    }

    if (check_attr.size() == 0) {
        int occupied = 0;
        int unoccupied = 0;
        string major;

        for (int k = 0; k < train_data.size(); k++) {
            train_data[k].get_target_class() == "1" ? occupied++ : unoccupied++;
        }

        (occupied > unoccupied) ? major = "1" : major = "0";

        create_leaf_node(p, major);
        return;
    }

    int     i, j;
    int     size = train_data.size();
    bool    leaf = true;
    string  target_class = train_data[0].get_target_class();
    for (i = 1; i < size; i++) {
        if (train_data[i].get_target_class() != target_class) {
            leaf = false;
            break;
        }
    }

    if (leaf) {
        create_leaf_node(p, target_class);
        return;
    }

    double  max_gain = -1;
    double  curr_gain = -1;
    int     max_index = 0;
    bool    is_continuous;
    double  divider;

    for (i = 0; i < check_attr.size(); i++) {
        if (attribute_type[check_attr[i]] == "continuous") {
            pair <double, double> temp = continuous_info_gain( train_data, check_attr[i]);
            curr_gain = temp.first;
            if (curr_gain > max_gain) {
                max_gain = curr_gain;
                max_index = i;
                is_continuous = true;
                divider = temp.second;
            }
        } else {
            curr_gain = discrete_info_gain( train_data, check_attr[i]);
            if (curr_gain > max_gain) {
                max_gain = curr_gain;
                max_index = i;
                is_continuous = false;
            }
        }
    }

    string attr_name = check_attr[max_index];
    check_attr.erase(check_attr.begin() + max_index);

    p = new decision_tree_node;
    if (is_continuous) {
        vector <string> values;
        p->set_type("continuous");
        p->set_attribute_name(attr_name);
        p->set_divider(divider);
        p->add_value("<=" + to_string(divider));
        p->add_value(">" + to_string(divider));

        values = p->get_values();
        for (i = 0; i < values.size(); i++) {
            string major;
            int occupied = 0;
            int unoccupied = 0;
            vector <example> sub_train_data;
            if( values[i].find("<=") != string::npos) {
                for (j = 0; j < train_data.size(); j++) {
                    train_data[j].get_target_class() == "1" ? occupied++ : unoccupied++;
                    if ((double)atof(train_data[j][attr_name].c_str()) <= divider){
                        sub_train_data.push_back(train_data[j]);
                    }
                }
            } else {
                for (j = 0; j < train_data.size(); j++) {
                    train_data[j].get_target_class() == "1" ? occupied++ : unoccupied++;
                    if ((double)atof(train_data[j][attr_name].c_str()) > divider){
                        sub_train_data.push_back(train_data[j]);
                    }
                }
            }
            (*p)[values[i]] = NULL;
            (occupied > unoccupied) ? major = "1" : major = "0";
            build(sub_train_data, (*p)[values[i]], check_attr, major);
        }
    } else {
        vector <string> values;
        p->set_type("discrete");
        p->set_attribute_name(attr_name);
        p->set_divider(0);

        for (i = 0; i < possible_values[attr_name].size(); i++) {
            p->add_value(possible_values[attr_name][i]);
        }

        values = p->get_values();
        for (i = 0; i < values.size(); i++) {
            string major;
            int occupied = 0;
            int unoccupied = 0;
            vector <example> sub_train_data;
            for (j = 0; j < train_data.size(); j++) {
                train_data[j].get_target_class() == "1" ? occupied++ : unoccupied++;
                if (train_data[j][attr_name] == values[i]) {
                    sub_train_data.push_back(train_data[j]);
                }
            }
            (*p)[values[i]] = NULL;
            (occupied > unoccupied) ? major = "1" : major = "0";
            build(sub_train_data, (*p)[values[i]], check_attr, major);
        }
    }
}


string decision_tree::classify (example &example, decision_tree_node *&p) {
    string predicted_class;
    string type = p->get_type();

    if (type == "leaf") {
        return p->get_attribute_name();
    }

    if (type == "continuous") {
        string              attribute_name = p->get_attribute_name();
        double              divider = p->get_divider();

        if (atof(example[p->get_attribute_name()].c_str()) > divider) {
            predicted_class = classify(example, (*p)[">" + to_string(divider)]);
        } else {
            predicted_class = classify(example, (*p)["<=" + to_string(divider)]);
        }
    } else {
        int                 i;
        string              attribute_name = p->get_attribute_name();
        vector <string>     values = p->get_values();

        for (i = 0; i < values.size(); i++) {
            if (example[p->get_attribute_name()] == values[i]) {
                predicted_class = classify(example, (*p)[values[i]]);
                break;
            }
        }
    }
    return predicted_class;
}

map <string, double> decision_tree::test (vector <example> &test_data) {
    int     i;
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
    string  predicted_class;
    int     total = test_data.size();

    map <string, double> stats;

    for (i = 0; i < test_data.size(); i++) {
        target_class = test_data[i].get_target_class();
        predicted_class = classify(test_data[i], root);

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

bool compare_example(example ex1, example ex2) {
    map <string, string> l = ex1.examples_list;
    map <string, string> r = ex2.examples_list;

    map <string, string>::iterator i, j;

    for (i = l.begin(), j = r.begin();
            i != l.end(), j != r.end(); i++, j++) {
        if (i->first.compare(j->first) != 0 || stod(i->second) != stod(j->second)) {
            return false;
        }
    }

    return true;

}

#define inc_example_weight(__example_idx_map, __weight_map, __example) {    \
    for (auto __it = __example_idx_map.begin();                             \
            __it != __example_idx_map.end(); __it++) {                      \
        if(compare_example(__example, __it->second)) {                      \
            __weight_map[__it->first]++;                                    \
            break;                                                          \
        }                                                                   \
    }                                                                       \
}

double decision_tree::test_rf (vector <example>       &boosted_data,
                             map <int, example>     &train_data_idx,
                             map <int, int>         &weighted_train_data ) {

    int     i;
    int     total = boosted_data.size();
    int     y_true = 0;
    string  target_class;
    string  predicted_class;

    for (i = 1; i < boosted_data.size(); i++) {
        int     idx;
        target_class = boosted_data[i].get_target_class();
        predicted_class = classify(boosted_data[i], root);

        if (target_class != predicted_class) {
            inc_example_weight(train_data_idx, weighted_train_data, boosted_data[i]);
        } else {
            y_true++;
        }
    }

    return (double)y_true / (double)total;
}

void decision_tree::print() {
    cout << endl;
    print(root, "", "");
}

void decision_tree::print(decision_tree_node *p, string prefix, string value) {
    cout << prefix;
    prefix += '\t';
    cout << (p->get_attribute_name()) << "(" << value << ")" << endl;
    vector <decision_tree_node *> children;
    vector <string> values;
    children = p->get_children_pointers();
    values = p->get_values();
    cout << prefix << "Children" << endl;
    for (int i = 0; i < children.size(); i++) {
        print(children[i], prefix, values[i]);
    }
}
