#include "utilities.h"

namespace stitcher3d
{
namespace utilities
{

std::vector<std::string> split_string(const std::string& str, const char delim)
{
    std::vector<std::string> ret_str;
    std::string current_str;
    for (auto& s : str)
    {
        if (s == delim)
        {
            ret_str.push_back(current_str);
            current_str.clear();
            continue;
        }
        
        current_str += s;
    }
    if (current_str != "")
        ret_str.push_back(current_str);

    return ret_str;
}

template <typename T>
std::vector<T> type_parse_str_vector(const std::vector<std::string>& str_vec)
{
    std::vector<T> parsed_vec;
    for (auto& s : str_vec)
    {
        if (typeid(T) == typeid(int))
            parsed_vec.push_back(std::stoi(s));
        else if (typeid(T) == typeid(float))
            parsed_vec.push_back(std::stof(s));
        else if (typeid(T) == typeid(double))
            parsed_vec.push_back(std::stod(s));

    }

    return parsed_vec;
}

o3d::camera::PinholeCameraIntrinsic read_camera_intrinsics(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;

    Eigen::Matrix3d intr_matrix;
    int width, height;
    
    int row_count = 0;
    if (file.is_open())
    {
        while (getline(file, line))
        {
            std::vector<std::string> split_row = utilities::split_string(line, ';');

            if (split_row.size() == 2)
            {
                std::vector<int> int_parsed = utilities::type_parse_str_vector<int>(split_row);
                width = int_parsed[0];
                height = int_parsed[1];
            }
            else if (split_row.size() == 3)
            {
                std::vector<double> float_parsed = utilities::type_parse_str_vector<double>(split_row);

                for (int i = 0; i < 3; i++)
                    intr_matrix(row_count, i) = float_parsed[i];

                row_count++;
            }
        }
    }
    else
        throw std::runtime_error("failed to read file! " + filename);

    file.close();

    auto intr = o3d::camera::PinholeCameraIntrinsic(width, height, intr_matrix.coeff(0, 0), 
                                intr_matrix.coeff(1, 1), intr_matrix.coeff(0, 2), intr_matrix.coeff(1, 2));

    return intr;
}

}
}