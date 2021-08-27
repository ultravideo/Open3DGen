#pragma once

#include <Open3D/Open3D.h>
#include <vector>
#include <array>
#include <algorithm>
#include <exception>
#include <memory>
#include "constants.h"

namespace stitcher3d
{

    template <typename T>
    struct pixel
    {
        T m_r, m_g, m_b, m_a;

        int m_channels;

        pixel() = delete;
        
        pixel(T r, T g, T b) :
            m_r(r), m_g(g), m_b(b), m_a(1) { m_channels = 3; }

        pixel(T r, T g, T b, T a) :
            m_r(r), m_g(g), m_b(b), m_a(a) { m_channels = 4; }

        pixel(T a) :
            m_r(0), m_g(0), m_b(0), m_a(a) { m_channels = 1; }

        

    };

    template <typename T>
    struct image
    {
        unsigned int m_width, m_height, m_channels;

        std::vector<std::vector<pixel<T>>> m_img_data;
        o3d::geometry::Image m_o3d_image;

        image() :
            m_width(0), m_height(0), m_channels(0) { }
        
        image(int width, int height, int channels) :
            m_width(width), m_height(height), m_channels(channels)
        {

        }

        /*
        image(std::vector<std::vector<T>> data, unsigned int width, 
                unsigned int height, unsigned int channels) :
            m_width(width), m_height(height),
            m_channels(channels), 
            m_img_data(data),
            m_o3d_image(o3d::geometry::Image) { }
        */

        /// format [r0, g0, b0], [r1, g1, b1], ...; as inner vector is width
        image(std::vector<T> data, unsigned int width, 
                unsigned int height, unsigned int channels) :
            m_width(width), m_height(height),
            m_channels(channels), 
            m_o3d_image(o3d::geometry::Image())
        {
            if (data.size() % (m_width * m_channels) != 0 || data.size() % (m_height * m_channels) != 0)
                throw std::runtime_error("invalid input data size, aborting in image constructor");

            for (int i = 0; i < (int)(data.size() / m_height) - 1; i++)
            {
                std::vector<pixel<T>> current_row;
                for (int j = 0; j < m_width; j++)
                {
                    const int data_i = i * (m_width * 3) + j * 3;
                    pixel<T> p(data[data_i], data[data_i + 1], data[data_i + 2]);
                    current_row.push_back(p);
                    // m_img_data.emplace_back(std::vector<T>(data.begin() + i * m_width, data.begin() + (i + 1) * m_width);
                }

                m_img_data.push_back(current_row);
            }
        }

        image(const o3d::geometry::Image& img) :
            m_width(img.width_), m_height(img.height_), m_channels(img.num_of_channels_)
        {
            const int bytes_per_ch = img.bytes_per_channel_;
            std::vector<uint8_t> img_data = img.data_;

            if (bytes_per_ch == 1 && m_channels == 3)
            {
                for (int i = 0; i < m_height; i++)
                {
                    std::vector<pixel<T>> current_row;

                    for (int j = 0; j < m_width; j++)
                    {
                        const int data_i = i * (m_width * 3) + j * 3;
                        pixel p((T)img_data[data_i], (T)img_data[data_i + 1], (T)img_data[data_i + 2]);
                        current_row.push_back(p);
                    }

                    m_img_data.push_back(current_row);
                }
            }
            else if (bytes_per_ch == 2 && m_channels == 1)
            {
                // open3d stores all images as uint8_t buffers, so this is fine
                for (int i = 0; i < m_height; i++)
                {
                    std::vector<pixel<T>> current_row;

                    for (int j = 0; j < m_width; j++)
                    {
                        const int data_i = i * (m_width * 3) + j * 3;
                        pixel p((T)img_data[data_i], (T)img_data[data_i + 1], (T)img_data[data_i + 2]);
                        current_row.push_back(p);
                    }

                    m_img_data.push_back(current_row);
                }
            }
            else
            {
                throw std::runtime_error("unknowon bytes per channel " +
                    std::to_string(bytes_per_ch) + " and m_channel " + std::to_string(m_channels) +
                    " aborting in image constructor on o3d::Image");
            }
        }

        // pixel<T>& operator , (int x, int y)
        // {
        //     return m_img_data[y][x];
        // }

        // const pixel<T>& operator , (int x, int y) const
        // {
        //     return m_img_data[y][x];
        // }

        /*
        void convert_to_o3d()
        {
            this->m_o3d_image = std::make_shared<o3d::geometry::Image>(o3d::geometry::Image());
            m_o3d_image->width_ = m_width;
            m_o3d_image->height_ = m_height;
            m_o3d_image->num_of_channels_ = m_channels;
            m_o3d_image->bytes_per_channel_ = sizeof(T);
            m_o3d_image->data_ = *m_img_data;
        }
        */
    };


}
