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
typedef bg::model::polygon<bg::model::d2::point_xy<double>,true,false> Polygon;
typedef bg::model::polygon<bg::model::d2::point_xy<double>,false,false> Polygon_cc;

PillarPoint::PillarPoint(double x,double y,double z,double r,
                         double canvas_x,double canvas_y)
{
    /*
        PillarPoint constructor. Sets the coordinates of the pillar point
        and the xp,yp variables which are the xy distance from the point to
        the pillar location. 
    */
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
    /*
        makes a single feature for the pillar point. The pybind11
        tensor to be filled in is passed in as input. Each element 
        of the feature is stored as a class attribute. 
    */

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
    /*
        Pillar constructor - sets the xy location
        of the pillar. 
    */

    this->canvas_x = canvas_x;
    this->canvas_y = canvas_y;
}

void Pillar::add_point(PillarPoint *pp){
    /*
        add a PillarPoint to the Pillar
    */

    this->points.push_back(pp);
}

std::vector<PillarPoint*> &Pillar::get_points(){
    return this->points;
}

std::vector<double> Pillar::point_mean() const{

    /*
        computes the mean of the x,y,z coordinates 
        of all the points in the pillar. 
    */

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
    
    /*
        computes the ious of a ground truth box and an anchor box.
        Polygon_cc is a polygon type where the corners are passed in
        counter clockwise order. Polygon is a polygon type where the
        points are passed in clockwise order.

        program exits if an iou < 0. this likely means that the ordering
        of the box corners is incorrect. 

    */

    Polygon_cc anchor{{{anchor_corners.at(anchor_index,0,0),anchor_corners.at(anchor_index,0,1)},
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
        std::cout << "IOU < 0 " << gt_index << std::endl;
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
        ort = 0;
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
        fills the pybind11 array `tensor` with the features
        from the lidar points in `points`. `indices` is filled
        with the xy location of the features in order to scatter
        back the tensor into a pseudo-image after the feature net.
    */

    /*
        create two hash maps, one will map from the xy pillar
        coordinates to the Pillar object, the other from the 
        xy pillar coordinates to the mean of the points in the 
        pillar
    */
    boost::unordered_map<boost::array<double,2>,Pillar*> pillar_map;
    boost::unordered_map<boost::array<double,2>,double*> means_map;

    // loop through each lidar point
    for (int i=0; i < points.shape()[0];i++)
    {
        // ignore if point is outside preset area
        if ((points.at(i,0) >= x_max) || (points.at(i,0) < x_min) || \
            (points.at(i,1) >= y_max) || (points.at(i,1) < y_min) || \
            (points.at(i,2) >= z_max) || (points.at(i,2) < z_min)){
            continue;
        }

        // compute which pillar the point belongs to 
        double canvas_x = floor((points.at(i,0) - x_min)/x_step);
        double canvas_y = floor((points.at(i,1) - y_min)/y_step);
        canvas_y = (canvas_height - 1) - canvas_y;
        // make new pillar point 
        PillarPoint *pp = new PillarPoint(points.at(i,0),points.at(i,1),
                                          points.at(i,2),points.at(i,3),
                                          canvas_x,canvas_y);
        
        boost::array<double,2> canvas;
        canvas[0] = canvas_x;
        canvas[1] = canvas_y;

        // if no Pillar object exits at canvas_x,canvas_y make a new
        // one, add the point to it and add it to the hash map
        if (pillar_map.find(canvas) == pillar_map.end())
        {
            Pillar *pillar = new Pillar(canvas_x,canvas_y);
            pillar->add_point(pp);
            pillar_map.insert({canvas,pillar});
        }
        // otherwise insert the point to the pillar
        else
        {
            Pillar *pillar = pillar_map.at(canvas);
            pillar->add_point(pp);
        }

        /*
            this computes the running mean of the points
            in the pillar - this is redundant, this 
            code should be moved inside the pillars
            class. 
        */
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

    int num_pillars = 0;
    boost::unordered::unordered_map<boost::array<double,2>,Pillar*>::iterator it;

    // loop through each pillar
    for (it = pillar_map.begin();it!=pillar_map.end(); ++it)
    {
        // if we hit the max number of pillars free the remaining memory
        // in the means, pillars and pillar points.
        if (num_pillars >= max_pillars){
            
            for (auto p: (it->second)->get_points()){
                delete p;
            }
            delete it->second;
            delete means_map.at(it->first);
            ++it;
            while (it !=pillar_map.end()){
                for (auto p: (it->second)->get_points()){
                    delete p;
                }
            delete it->second;
            delete means_map.at(it->first);
            ++it;
            }
            
            break;
        }

        boost::array<double,2> canvas = it->first;
        Pillar *pillar = it->second; 

        double *pillar_mean = means_map.at(canvas);
        int num_points = 0;
        std::vector<PillarPoint*> pillar_points = pillar->get_points();

        // loop through each of the points in the pillar 
        for (int i =0; i < pillar_points.size(); i++)
        {
            // if we hit the max number of points in a pillar
            // free the memory from the remaining points
            if (num_points >= max_points_per_pillar){
                while (i < pillar_points.size()){
                    delete pillar_points[i];
                    i++;
                }
                break;
            }

            // make the feature from the point in the pillar
            PillarPoint *p = pillar_points[i];
            p->set_xc(pillar_mean[0] - p->get_x());
            p->set_yc(pillar_mean[1] - p->get_y());
            p->set_zc(pillar_mean[2] - p->get_z());
            p->make_feature(tensor,num_pillars,num_points);
            num_points++;
            delete p;
        }
        // set the indices which map the pillar number 
        // to its xy canvas location
        indices.mutable_at(num_pillars,0) = 1;
        indices.mutable_at(num_pillars,1) = canvas[0];
        indices.mutable_at(num_pillars,2) = canvas[1];
        num_pillars++;
        delete pillar_mean;
        delete it->second;
    }
    
}

void make_ious(py::array_t<double> &a_corners,
               py::array_t<double> &g_corners,
               py::array_t<double> &a_centers,
               py::array_t<double> &g_centers,
               py::array_t<double> &ious)
            
{    
    /*
        make ious for all anchors and gt boxes passed in 
        through `a_corners` & `g_corners`

        since doing the iou computation is expensive we can 
        cut down a lot of time by ignoring boxes whose centers
        are far apart.
    */

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
