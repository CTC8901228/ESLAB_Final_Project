
import copy
import open3d as o3d
import numpy as np
import math
"""
this is the method of normalized icp
source:query numpy array with size=(# point,3)
target:target numpy array with size=(# point,3)
threshold:threshold of icp
"""
def icp(source,target,threshold=999):
    # source = np.random.rand(1000,3)*100#o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    # target =np.random.rand(1000,3)*100 #o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    n_source=source.shape[0]
    n_target=target.shape[0]
    
    source_vec_mean=source-np.mean(source,0)  ##vec with centroid
    source_dist_mean=np.sum( np.sqrt( np.sum(source_vec_mean* source_vec_mean,1)) )/n_source
    source=source_vec_mean/source_dist_mean
    
    
    target_vec_mean=target-np.mean(target,0)  ##vec with centroid
    target_dist_mean= np.sum( np.sqrt(np.sum(target_vec_mean* target_vec_mean,1)) )/n_target
    target=target_vec_mean/target_dist_mean
    # source_vec_mean=source-np.mean(source,0)  ##vec with centroid
    # source_dist_mean= math.sqrt( np.sum(source_vec_mean* source_vec_mean) )/n_source
    # target_vec_mean=target-np.mean(target,0)  ##vec with centroid
    # target_dist_mean= math.sqrt( np.sum(target_vec_mean* target_vec_mean) )/n_target
    # print(source_vec_mean* source_vec_mean )
    # print( source_vec_mean )
    # # print(target_dist_mean)
    
    # print(source)
    # print(target)
    pcd_source=o3d.geometry.PointCloud()
    pcd_target=o3d.geometry.PointCloud()
    
    pcd_source.points=o3d.utility.Vector3dVector(source)
    pcd_target.points=o3d.utility.Vector3dVector(target)
    source=pcd_source
    target=pcd_target
    # threshold = 2
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4],
                         [0.0, 0.0, 0.0, 1.0]])
    # draw_registration_result(source, target, trans_init)
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 8))
    print((reg_p2p.inlier_rmse))
    
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])



if __name__ == '__main__':
    source1 = np.random.rand(1000,3)#o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    source2 = np.random.rand(1000,3)*100+np.array([[i,i,i] for i in range(1000)])#o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target =np.random.rand(800,3)+np.array([[i,i,i] for i in range(800)])
    icp(source1,target)
    icp(source2,target)
