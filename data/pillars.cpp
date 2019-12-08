#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <stdlib.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/unordered_map.hpp>
#include <boost/array.hpp>
#include <math.h>
#include "pillars.h"

namespace py = pybind11;
namespace bg = boost::geometry;
typedef bg::model::polygon<bg::model::d2::point_xy<double>,false,false> Polygon;

PillarPoint::PillarPoint(double x,double y,double z,double r,
                         double canvas_x,double canvas_y)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->r = r;
    this->xp = canvas_x - x;
    this->yp = canvas_y - y;
    this->xc = 0;
    this->yc = 0;
    this->zc = 0;

}

void PillarPoint::make_feature(py::array_t<double> &tensor,
                               int pillar_ind,
                               int point_ind) const 
{
    tensor.mutable_at(pillar_ind,point_ind,0) = this->x;
    tensor.mutable_at(pillar_ind,point_ind,1) = this->y;
    tensor.mutable_at(pillar_ind,point_ind,2) = this->z;
    tensor.mutable_at(pillar_ind,point_ind,3) = this->r;
    tensor.mutable_at(pillar_ind,point_ind,4) = this->xp;
    tensor.mutable_at(pillar_ind,point_ind,5) = this->yp;
    tensor.mutable_at(pillar_ind,point_ind,6) = this->xc;
    tensor.mutable_at(pillar_ind,point_ind,7) = this->yc;
    tensor.mutable_at(pillar_ind,point_ind,8) = this->zc;


}

void PillarPoint::set_xc(double xc){
    this->xc = xc;
}
void PillarPoint::set_yc(double yc){
    this->yc = yc;
}
void PillarPoint::set_zc(double zc){
    this->zc = zc;
}

double PillarPoint::get_x() const{
    return this->x;
}

double PillarPoint::get_y() const{
    return this->y;
}

double PillarPoint::get_z() const{
    return this->z;
}

Pillar::Pillar(double canvas_x,double canvas_y){
    this->canvas_x = canvas_x;
    this->canvas_y = canvas_y;
}

void Pillar::add_point(PillarPoint *pp){
    this->points.push_back(pp);
}

std::vector<PillarPoint*> &Pillar::get_points(){
    return this->points;
}

std::vector<double> Pillar::point_mean() const{
    double x_total = 0;
    double y_total = 0;
    double z_total = 0;
    double num_points = (double) this->points.size();
    std::vector<double> coord_means;

    for (auto p : this->points){
        x_total += p->get_x();
        y_total += p->get_y();
        z_total += p->get_z();
    }

    coord_means.push_back(x_total/num_points);
    coord_means.push_back(y_total/num_points);
    coord_means.push_back(z_total/num_points);
    
    return coord_means;

}

double iou(py::array_t<double> &anchor_corners,
           py::array_t<double> &gt_corners,
           int anchor_index,
           int gt_index)
{
    
    Polygon anchor{{{anchor_corners.at(anchor_index,0,0),anchor_corners.at(anchor_index,0,1)},
                   {anchor_corners.at(anchor_index,1,0),anchor_corners.at(anchor_index,1,1)},
                   {anchor_corners.at(anchor_index,2,0),anchor_corners.at(anchor_index,2,1)},
                   {anchor_corners.at(anchor_index,3,0),anchor_corners.at(anchor_index,3,1)}}};
    
    Polygon gt_box{{{gt_corners.at(gt_index,0,0),gt_corners.at(gt_index,0,1)},
                   {gt_corners.at(gt_index,1,0),gt_corners.at(gt_index,1,1)},
                   {gt_corners.at(gt_index,2,0),gt_corners.at(gt_index,2,1)},
                   {gt_corners.at(gt_index,3,0),gt_corners.at(gt_index,3,1)}}};

    std::vector<Polygon> output;
    bg::intersection(anchor,gt_box,output);
    if (output.size() == 0){
        return 0;
    }
    double int_area = bg::area(output[0]);
    double iou = int_area/(bg::area(anchor) + bg::area(gt_box) - int_area);
    if (iou < 0){
        std::cout << "IOU < 0 " << std::endl;
        std::exit(1); 
    }
    return iou;    

}

void make_target(py::array_t<double> &a_ctrs,
                 py::array_t<double> &gt_ctrs,
                 py::array_t<double> &a_wlh,
                 py::array_t<double> &gt_wlh,
                 py::array_t<double> &a_yaws,
                 py::array_t<double> &gt_yaws,
                 py::array_t<double> &gt_labels,
                 py::array_t<double> &targets,
                 int a_ind,
                 int gt_ind,
                 int target_ind)
{
    double ax = a_ctrs.at(a_ind,0);
    double ay = a_ctrs.at(a_ind,1);
    double az = a_ctrs.at(a_ind,2);

    double gx = gt_ctrs.at(gt_ind,0);
    double gy = gt_ctrs.at(gt_ind,1);
    double gz = gt_ctrs.at(gt_ind,2);

    double aw = a_wlh.at(a_ind,0);
    double al = a_wlh.at(a_ind,1);
    double ah = a_wlh.at(a_ind,2);

    double gw = gt_wlh.at(gt_ind,0);
    double gl = gt_wlh.at(gt_ind,1);
    double gh = gt_wlh.at(gt_ind,2);

    double at = a_yaws.at(a_ind);
    double gt = gt_yaws.at(gt_ind);

    double ad = sqrt(aw*aw + al*al);

    double dx = (gx - ax)/ad;
    double dy = (gy - ay)/ad;
    double dz = (gz - az)/ah;

    double dw = log(gw/aw);
    double dl = log(gl/al);
    double dh = log(gh/ah);

    double dt = sin(gt - at);
    double class_ind = gt_labels.at(gt_ind);

    double ort = 1;
    if (gt < 0){
        ort = -1;
    }

    targets.mutable_at(target_ind,0) = 1;
    targets.mutable_at(target_ind,1) = dx;
    targets.mutable_at(target_ind,2) = dy;
    targets.mutable_at(target_ind,3) = dz;
    targets.mutable_at(target_ind,4) = dw;
    targets.mutable_at(target_ind,5) = dl;
    targets.mutable_at(target_ind,6) = dh;
    targets.mutable_at(target_ind,7) = dt;
    targets.mutable_at(target_ind,8) = ort;
    targets.mutable_at(target_ind,9) = class_ind;

}


void create_pillars(py::array_t<double> &points,
                    py::array_t<double> &tensor,
                    py::array_t<double> &indices,
                    int max_points_per_pillar,
                    int max_pillars,
                    double x_step,
                    double y_step,
                    double x_min,
                    double y_min,
                    double z_min,
                    double x_max,
                    double y_max,
                    double z_max,
                    double canvas_height)
{
    /*
    std::cerr << "CREATE PILLARS\n";
    boost::unordered_map<boost::array<double,2>,Pillar*> pillar_map;
    boost::unordered_map<boost::array<double,2>,double*> means_map;

    for (int i=0; i < points.shape()[0];i++)
    {
        if ((points.at(i,0) >= x_max) || (points.at(i,0) < x_min) || \
            (points.at(i,1) >= y_max) || (points.at(i,1) < y_min) || \
            (points.at(i,2) >= z_max) || (points.at(i,2) < z_min)){
            continue;
        }
        std::cout << "processing point:\n";
        std::cout << points.at(i,0) << ", " << points.at(i,1) << ", " << points.at(i,2);
        double canvas_x = floor((points.at(i,0) - x_min)/x_step);
        double canvas_y = floor((points.at(i,1) - y_min)/y_step);
        canvas_y = (canvas_height - 1) - canvas_y;
        PillarPoint *pp = new PillarPoint(points.at(i,0),points.at(i,1),
                                          points.at(i,2),points.at(i,3),
                                          canvas_x,canvas_y);
        
        boost::array<double,2> canvas;
        canvas[0] = canvas_x;
        canvas[1] = canvas_y;
    
        if (pillar_map.find(canvas) == pillar_map.end())
        {
            Pillar *pillar = new Pillar(canvas_x,canvas_y);
            pillar->add_point(pp);
            pillar_map.insert({canvas,pillar});
        }
        else
        {
            Pillar *pillar = pillar_map.at(canvas);
            pillar->add_point(pp);
        }

        if (means_map.find(canvas) == means_map.end())
        {
            double *means = new double[4];
            means[0] = points.at(i,0);
            means[1] = points.at(i,1);
            means[2] = points.at(i,2);
            means[3] = 1;
            means_map.insert({canvas,means});
        }
        else
        {
            double *means = means_map.at(canvas);
            double n = means[3];
            means[0] = means[0]*(n/(n+1)) + points.at(i,0)/(n+1);
            means[1] = means[1]*(n/(n+1)) + points.at(i,1)/(n+1);
            means[2] = means[2]*(n/(n+1)) + points.at(i,2)/(n+1);
            means[3] = n+1;
        }
    }

    std::cerr << "done assigning points to pillars\n";
    int num_pillars = 0;
    boost::unordered::unordered_map<boost::array<double,2>,Pillar*>::iterator it;
    for (it = pillar_map.begin();it!=pillar_map.end(); ++it)
    {
        if (num_pillars >= max_pillars){
            /*
            std::cerr << "too many pillars\n";
            for (auto p: (it->second)->get_points()){
                delete p;
            }
            delete it->second;
            while (it !=pillar_map.end()){
                for (auto p: (it->second)->get_points()){
                    delete p;
                }
            delete it->second;
            ++it;
            }
            
            break;
        }

        boost::array<double,2> canvas = it->first;
        Pillar *pillar = it->second; 

        double *pillar_mean = means_map.at(canvas);
        int num_points = 0;
        std::vector<PillarPoint*> pillar_points = pillar->get_points();
        for (int i =0; i < pillar_points.size(); i++)
        {
            if (num_points >= max_points_per_pillar){
                /*
                std::cerr << "too many points\n";
                while (i < pillar_points.size()){
                    delete pillar_points[i];
                    i++;
                }
                
                break;
            }
            PillarPoint *p = pillar_points[i];
            p->set_xc(pillar_mean[0] - p->get_x());
            p->set_yc(pillar_mean[1] - p->get_y());
            p->set_zc(pillar_mean[2] - p->get_z());
            p->make_feature(tensor,num_pillars,num_points);
            num_points++;
            std::cerr << "freeing pillar point p \n";
            //delete p;
        }
        indices.mutable_at(num_pillars,0) = 1;
        indices.mutable_at(num_pillars,1) = canvas[0];
        indices.mutable_at(num_pillars,2) = canvas[1];
        num_pillars++;
        std::cerr << "freeing pillar mean\n";
        //delete pillar_mean;
        //delete it->second;
    }
    */
    int x = 1;
}

void make_ious(py::array_t<double> &a_corners,
               py::array_t<double> &g_corners,
               py::array_t<double> &a_centers,
               py::array_t<double> &g_centers,
               py::array_t<double> &ious)
            
{    
    for (int i=0; i < a_corners.shape()[0];i++){
        for(int j=0; j < g_corners.shape()[0];j++){
            if ((std::abs(a_centers.at(i,0) - g_centers.at(j,0)) > 10) || \
                (std::abs(a_centers.at(i,1) - g_centers.at(j,1)) > 10))
            {
                ious.mutable_at(i,j) = 0;
                continue;
            }
            ious.mutable_at(i,j) = iou(a_corners,g_corners,i,j);
        }
    }
}

PYBIND11_MODULE(pillars,m)
{
    m.doc() = "point pillars data prep functions";
    m.def("make_ious",&make_ious,"ious");
    m.def("create_pillars",&create_pillars,"pillars");

}
