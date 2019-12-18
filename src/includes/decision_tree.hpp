#ifndef _DECISION_TREE_H_
#define _DECISION_TREE_H_

#include <set>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class reader {
    public:
        static vector <vector <string> > read_data( string file );
        static set <string> read_target_value( string file, int n );
};

class instance {
    public:
        instance();
        instance( const vector <string> &attr_names,
                  const vector <string> &attr_values );
        string operator[]( const string &attr_name );
        map <string, string> examples_list;
};

class example : public instance {
    public:
        example();
        example( const vector <string>  &attr_names,
                 const vector <string>  &attr_values,
                 const string           &target_class );
        // bool operator==(example ex);
        string get_target_class() const;

    private:
        string target_class;
};

class decision_tree_node {
    public:
        decision_tree_node();
        void set_attribute_name( const string &attr_name );
        void set_type( const string &type );
        void set_divider( const double &divider );
        void add_value( const string &value);
        string get_attribute_name() const;
        string get_type() const;
        double get_divider() const;
        vector <string> get_values() const;

        decision_tree_node *& operator[]( const string &attr_value );
        vector <decision_tree_node *> get_children_pointers();

    private:
        string                                          attr_name;
        string                                          type;
        double                                          divider;
        vector <string>                                 values;
        unordered_map <string, decision_tree_node *>    children;
};

class decision_tree {
    public:
        void add_attr_info( const string            &attr_name,
                            const vector <string>   &attr_values,
                            int                     attr_type );
        void add_target_values( set <string> target_values );
        void build( const vector <example> &train_data );
        map <string, double> test (vector <example> &test_data);
        double test_rf (vector <example>      &boosted_data,
                      map <int, example>    &train_data_idx,
                      map <int, int>        &weighted_train_data);
        void print();
        string classify (example &example, decision_tree_node *&p);

        decision_tree_node                          *root;

    private:
        double calculate_entropy( const map <string, int> &entropy_map);
        double discrete_info_gain( vector <example> &examples_list,
                                   const string     &attr_name );
        pair <double, double> continuous_info_gain(
                    vector <example> &examples_list,
                    const string     &attr_name );
        void build( vector <example>    train_data,
                    decision_tree_node  *&p,
                    vector <string>     check_attr,
                    string              majority );
        void print( decision_tree_node *p, string prefix, string value);

        unordered_map <string, vector <string>>     possible_values;
        unordered_map <string, string>              attribute_type;
        set <string>                                target_values;
};

#endif
