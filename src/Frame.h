/*
 * Frame.h
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#ifndef FRAME_H_
#define FRAME_H_

#include "Camera.h"
#include "common_include.h"

namespace myfusion {

class Frame {
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long                  id;         // id of this frame
    double                         time_stamp; // when it is recorded
    SE3                            T_w_c;      // transform from camera to world
    Camera::Ptr                    camera;     // Pinhole RGBD Camera model
    unsigned char*                 depth_data;
    unsigned char*                 texture_data;
    bool                           is_key_frame;  // whether a key-frame
    unsigned int                   width_depth;
    unsigned int 				   height_depth;
    unsigned int 			       width_texture;
    unsigned int 				   height_texture;

public: // data members
    Frame();
    ~Frame();

    unsigned char * getDepthData() {return depth_data;}
    unsigned char * getTextureData() {return texture_data;}

    void CreateNewFrame(Camera::Ptr& camera, Mat& depth, Mat& texture);
};

} /* namespace myfusion */

#endif /* FRAME_H_ */
