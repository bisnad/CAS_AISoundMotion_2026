import sys

from fbx import *
from common import FbxCommon

import numpy as np


class FBX_Skeleton_Handler:
    def __init__(self):
        
        self.root_node = None
        
        self.joints = []
        self.parents = []
        self.children = []
        self.joint_offsets = []

class FBX_Mocap_Data:
    def __init__(self):
        
        self.skeleton_root_node = None
        self.skeleton_nodes = []
        
        self.skeleton_root = None
        self.skeleton_joints = []
        self.skeleton_parents = []
        self.skeleton_children = []
        self.skeleton_joint_offsets = []

        self.motion_frame_rate = 0
        self.motion_rot_sequence = []
        
        self.motion_times = {} # ADDED
        self.motion_pos_local = []
        self.motion_rot_local_euler = []


# NOTE: Time modes 
FBX_TimeModes = {
    "eDefaultMode" :    0,
    "eFrames120" :      120,
    "eFrames100" :      100,
    "eFrames60" :       60,
    "eFrames50" :       50,
    "eFrames48" :       48,
    "eFrames30" :       30,
    "eFrames30Drop" :   30,
    "eNTSCDropFrame" :  29.97002617,
    "eNTSCFullFrame" :  29.97002617,
    "ePAL" :            25,
    "eFrames24" :       24,
    "eFrames1000" :     1000,
    "eFilmFullFrame" :  24,
    "eCustom" :         "Custom",
    "eFrames96" :       96,
    "eFrames72" :       72,
    "eFrames59dot94" :  59.94005234
    }

class FBX_Tools:

    def __init__(self):
        self.reset()
        pass
    
    def reset(self):
        
        self.sdkManager = None
        self.scene = None

        self.animation_stacks = []
        self.animation_layers = []
        self.animation_layer = None
        
        self.mocap_data = []

        self.file_name = ""


    def load(self, mocap_file_path):
        self.reset()
        
        # Prepare FBX SDK
        self.sdkManager, self.scene = FbxCommon.InitializeSdkObjects()

        # Load Scene
        result = FbxCommon.LoadScene(self.sdkManager, self.scene, mocap_file_path)
        
        if(result == False):
            print("failed to load mocap file ", mocap_file_path)
            return
        
        # get skeleton root nodes
        skeletonRoots = FBX_Tools.getSkeletonRootNodes(self.scene)
        
        if(len(skeletonRoots) == 0):
            print("no skeleton roots found")
            return
        
        self.animation_stacks = FBX_Tools.getAnimationStacks(self.scene)
        
        if len(self.animation_stacks) == 0:
            print("no animation stacks found")
            return
        
        # gather all animation layers
        self.animation_layers = []
        for anim_stack in self.animation_stacks:
            anim_layers = FBX_Tools.getAnimationLayers(anim_stack)
            self.animation_layers += anim_layers
            
        if len(self.animation_layers) == 0:
            print("no animation layers found")
            return
        elif len(self.animation_layers) > 1:
            print("Warning: more than one animation layer found but only one layer supported at the moment")
            
        self.animation_layer = self.animation_layers[0]
        
        # create one Mocap Data instance per skeleton and populate it with some info
        for root in skeletonRoots:
            skel_mocap_data = FBX_Mocap_Data()
            skel_mocap_data.skeleton_root_node = root
            
            self.mocap_data.append(skel_mocap_data)
        
        # collect skeleton data for each mocap data
        for skel_mocap_data in self.mocap_data:
            
            skel_mocap_data.skeleton_root = skel_mocap_data.skeleton_root_node.GetName()
            skel_mocap_data.skeleton_nodes = FBX_Tools.getSkeletonNodes(skel_mocap_data.skeleton_root_node)
            
            skeleton_data = FBX_Tools.getSkeletonData(skel_mocap_data.skeleton_root_node)

            skel_mocap_data.skeleton_joints = skeleton_data["joints"]
            skel_mocap_data.skeleton_parents = skeleton_data["parents"]
            skel_mocap_data.skeleton_children = skeleton_data["children"]
            skel_mocap_data.skeleton_joint_offsets = skeleton_data["offsets"]
            
        # extract framerate
        frameRate = FBX_Tools.getFrameRate(self.scene)
        
        if frameRate < 0:
            print("Error: custom frame rate not supported")
            return None
        
        # NOTE: this works only for single animation stack
        frameCount = FBX_Tools.getFrameCount(self.scene, self.animation_stacks[0])
            
        # extract animation data
        for skel_mocap_data in self.mocap_data:
            
            skel_mocap_data.motion_frame_rate = frameRate
            skel_mocap_data.motion_rot_sequence = FBX_Tools.getRotationSequence(skel_mocap_data.skeleton_root_node)

            times_per_joint, pos_local, rot_local_euler = FBX_Tools.getMotion(skel_mocap_data.skeleton_nodes, self.animation_layer, frameCount, frameRate)

            skel_mocap_data.motion_times = times_per_joint
            skel_mocap_data.motion_pos_local = pos_local
            skel_mocap_data.motion_rot_local_euler = rot_local_euler

        # Destroy the FBX SDK manager
        self.sdkManager.Destroy()
        
        return self.mocap_data

    def write(self, mocap_data, fileName):
        
        self.reset()
        self.mocap_data = mocap_data
        
        # Prepare the FBX SDK
        self.sdkManager, self.scene = FbxCommon.InitializeSdkObjects()
        self.time = FbxTime()
        
        # don't ask me why these values
        FbxAnimCurveDef.sDEFAULT_WEIGHT = 1.0
        FbxAnimCurveDef.sDEFAULT_VELOCITY = 1.0
        
        # create skeletons, one for each skel_mocap_data
        self.fbx_skeletons = []
        for skel_mocap_data in self.mocap_data:
            fbx_skeleton = FBX_Tools.createSkeleton(self.scene, skel_mocap_data)
            self.fbx_skeletons.append(fbx_skeleton)
        
        # Add skeletons in the Scene
        self.fbx_root_node = self.scene.GetRootNode()
        for fbx_skeleton in self.fbx_skeletons:
            self.fbx_root_node.AddChild(fbx_skeleton.root)
            
        # List of all anim layers, in case there are multiple skeletons in the scene
        self.fbx_animLayers = {}
        
        for sI, (skeleton, skel_mocap_data) in enumerate(zip(self.fbx_skeletons, self.mocap_data)):
            self.fbx_animLayers[sI] = FBX_Tools.createAnimationLayer(self.scene, skeleton, skel_mocap_data, sI)
        
        # export scene
        FBX_Tools.exportFBX(fileName, self.sdkManager, self.scene)
        
            
    # Gather all skeleton root nodes
    @staticmethod
    def getSkeletonRootNodes(pScene):
        
        skeletonRoots = []
        sceneRoot = pScene.GetRootNode()

        if sceneRoot:
            FBX_Tools.findSkeletonRootNode(sceneRoot, skeletonRoots)
            
        return skeletonRoots
    
    # traverse node hiearchy until first skeleton nodes are found
    @staticmethod
    def findSkeletonRootNode(pNode, skelRootNodes):
        
        nodeAttribute = pNode.GetNodeAttribute()
        if nodeAttribute is not None:
            nodeAttributeType = (nodeAttribute.GetAttributeType())
            
            if nodeAttributeType is FbxNodeAttribute.EType.eSkeleton:
                skelRootNodes.append(pNode)
                return
        
        childCount = pNode.GetChildCount()
        for i in range(pNode.GetChildCount()):
            childNode = pNode.GetChild(i)
            FBX_Tools.findSkeletonRootNode(childNode, skelRootNodes)
        
    @staticmethod
    def getFrameRate(pScene):
        
        timeMode = pScene.GetGlobalSettings().GetTimeMode()
        fps_string = FBX_TimeModes[timeMode.name]
        
        try:
            fps_float = float(fps_string)
        except ValueError:
            fps_float = -1
            
        return fps_float

    # get animation stacks
    @staticmethod
    def getAnimationStacks(pScene):
        
        animationStacks = []
        
        for i in range(pScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))):
            
            animStack = pScene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
            animationStacks.append(animStack)
            
        return animationStacks

    # get animation layers
    @staticmethod
    def getAnimationLayers(pAnimStack):
        
        animationLayers = []

        for i in range(pAnimStack.GetMemberCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId))):
            
            animLayer = pAnimStack.GetMember(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), i)
            animationLayers.append(animLayer)
            
        return animationLayers

    # get number of frames from animation stack
    @staticmethod
    def getFrameCount(pScene, pAnimStack):
        
        lTimeSpan = pAnimStack.GetLocalTimeSpan()
        lTimeStart = lTimeSpan.GetStart()
        lTimeStop = lTimeSpan.GetStop()
        lTimeDiff = lTimeStop - lTimeStart
        
        globalTimeSettings = pScene.GetGlobalSettings().GetTimeMode()
        frameCount = lTimeDiff.GetFrameCount(globalTimeSettings)
        
        return frameCount
    
    # get rotation sequence
    @staticmethod
    def getRotationSequence(pNode):
        
        rotationOrder = pNode.RotationOrder.Get()
        
        return rotationOrder

    # set rotation sequence
    @staticmethod
    def setRotationSequence(pNode, pRotationOrder):
        
        pNode.RotationOrder.Set(pRotationOrder)

    # get skeleton nodes
    @staticmethod
    def getSkeletonNodes(pSkeletonRootNode):
        
        nodes = []
        FBX_Tools.traverseSkeletonNodes(pSkeletonRootNode, nodes)
        
        return nodes
    
    # recursively gather all nodes
    @staticmethod
    def traverseSkeletonNodes(pNode, pNodes):
        
        pNodes.append(pNode)
        
        childCount = pNode.GetChildCount()
        for i in range(childCount):
            childNode = pNode.GetChild(i)
            FBX_Tools.traverseSkeletonNodes(childNode, pNodes)
            
    # get node values from a single curve
    @staticmethod
    def getNodeValues(pNode, pCurve):
        
        keyTimes = []
        keyValues = []
        
        if pCurve is not None:
            keyCount = pCurve.KeyGetCount()
            for key in range(keyCount):
                lKeyValue = pCurve.KeyGetValue(key)
                lKeyTime  = pCurve.KeyGetTime(key)
                
                keyTimes.append(lKeyTime.GetSecondDouble())
                keyValues.append(lKeyValue)
            
        return keyTimes, keyValues
            
    # get local position and local rotation (euler) of a node
    @staticmethod
    def getNodeRotPos(pNode, pAnimLayer, pFrameCount, pFrameRate):
        
        posXCurve = pNode.LclTranslation.GetCurve(pAnimLayer, "X")
        posYCurve = pNode.LclTranslation.GetCurve(pAnimLayer, "Y")
        posZCurve = pNode.LclTranslation.GetCurve(pAnimLayer, "Z")
        
        rotXCurve = pNode.LclRotation.GetCurve(pAnimLayer, "X")
        rotYCurve = pNode.LclRotation.GetCurve(pAnimLayer, "Y")
        rotZCurve = pNode.LclRotation.GetCurve(pAnimLayer, "Z")
        
        posXTimes, posXValues = FBX_Tools.getNodeValues(pNode, posXCurve)
        posYTimes, posYValues = FBX_Tools.getNodeValues(pNode, posYCurve)
        posZTimes, posZValues = FBX_Tools.getNodeValues(pNode, posZCurve)
        
        rotXTimes, rotXValues = FBX_Tools.getNodeValues(pNode, rotXCurve)
        rotYTimes, rotYValues = FBX_Tools.getNodeValues(pNode, rotYCurve)
        rotZTimes, rotZValues = FBX_Tools.getNodeValues(pNode, rotZCurve)
        
        all_times = [posXTimes, posYTimes, posZTimes, rotXTimes, rotYTimes, rotZTimes]
        nodeTimes = []
        for t_arr in all_times:
            if len(t_arr) > len(nodeTimes):
                nodeTimes = t_arr
                
        if len(nodeTimes) == 0:
            safe_fps = pFrameRate if pFrameRate > 0 else 50.0
            nodeTimes = [float(i) / safe_fps for i in range(pFrameCount)]
            
        offset = pNode.LclTranslation.Get()
        
        if len(posXValues) == 0:
            posXValues = [offset[0]] * pFrameCount
        if len(posYValues) == 0:
            posYValues = [offset[1]] * pFrameCount
        if len(posZValues) == 0:
            posZValues = [offset[2]] * pFrameCount
            
        if len(rotXValues) == 0:
            rotXValues = [0] * pFrameCount
        if len(rotYValues) == 0:
            rotYValues = [0] * pFrameCount
        if len(rotZValues) == 0:
            rotZValues = [0] * pFrameCount

        nodePos = [posXValues, posYValues, posZValues]
        nodeRot = [rotXValues, rotYValues, rotZValues]

        return nodeTimes, nodePos, nodeRot

    # iterate through all nodes and get their animation curves
    @staticmethod
    def getMotion(pNodes, pAnimLayer, pFrameCount, pFrameRate):
        
        pos_local = []
        rot_local_euler = []    
        times_per_joint = {}

        for nI, node in enumerate(pNodes):
            
            nodeTimes, node_pos, node_rot_euler = FBX_Tools.getNodeRotPos(node, pAnimLayer, pFrameCount, pFrameRate)

            node_pos = np.transpose(np.array(node_pos))
            node_rot_euler = np.transpose(np.array(node_rot_euler))
            
            pos_local.append(node_pos)
            rot_local_euler.append(node_rot_euler)
            times_per_joint[nI] = np.array(nodeTimes)

        return times_per_joint, pos_local, rot_local_euler

    # get entire skeleton info
    @staticmethod
    def getSkeletonData(pRootNode):
        
        skeletonData = {}
        
        skeletonData["joints"] = []
        skeletonData["parents"] = []
        skeletonData["children"] = []
        skeletonData["offsets"] = []

        FBX_Tools.traverseSkeletonNodeHierarchy(None, pRootNode, skeletonData)

        # change from name to index based references for children
        for i, c in enumerate(skeletonData["children"]):
            c_indices = []
            for n in c:
                c_indices.append(skeletonData["joints"].index(n))
            skeletonData["children"][i] = c_indices
        
        # change from name to index based references for parents
        for i, p in enumerate(skeletonData["parents"]):
            if p == "":
                skeletonData["parents"][i] = -1
            else:
                skeletonData["parents"][i] = skeletonData["joints"].index(p)
                
        return skeletonData
            
    # recursively traverse the skeleton node hierarchy and gather info about joints and their relationships
    @staticmethod
    def traverseSkeletonNodeHierarchy(pParentNode, pNode, skeletonData):
        
        if pParentNode is None:
            parentName = ""
        else:
            parentName = pParentNode.GetName()
        skeletonData["parents"].append(parentName)
        
        jointName = pNode.GetName()
        skeletonData["joints"].append(jointName)
        
        tr = pNode.LclTranslation.Get()
        offset = [tr[0], tr[1], tr[2]]
        skeletonData["offsets"].append(offset)
        
        children = []
        childCount = pNode.GetChildCount()
        for i in range(childCount):
            childNode = pNode.GetChild(i)
            childName = childNode.GetName()
            children.append(childName)
        skeletonData["children"].append(children)
        
        for i in range(childCount):
            childNode = pNode.GetChild(i)
            FBX_Tools.traverseSkeletonNodeHierarchy(pNode, childNode, skeletonData)
            
    # ---- Methods for exporting FBX ----
            
    @staticmethod
    def createSkeleton(pScene, pMocap_data):
        
        fbx_skeleton_handler = FBX_Skeleton_Handler()
        
        skeleton_joints = pMocap_data.skeleton_joints
        skeleton_children = pMocap_data.skeleton_children
        joint_offsets = pMocap_data.skeleton_joint_offsets 
        rot_sequence = pMocap_data.motion_rot_sequence 
        
        fbx_reference_node = FbxNode.Create(pScene, ("Skeleton"))
        FBX_Tools.setRotationSequence(fbx_reference_node, rot_sequence)

        joint_count = len(skeleton_joints)
        for jI in range(joint_count):
            
            joint_name = skeleton_joints[jI]
            
            fbx_skeleton_node_attribute = FbxSkeleton.Create(pScene, joint_name)
            fbx_skeleton_node_attribute.SetSkeletonType(FbxSkeleton.EType.eLimbNode)
            fbx_skeleton_node_attribute.Size.Set(1.0)
            
            fbx_skeleton_node = FbxNode.Create(pScene, joint_name)
            fbx_skeleton_node.SetNodeAttribute(fbx_skeleton_node_attribute)
            FBX_Tools.setRotationSequence(fbx_skeleton_node, rot_sequence)
            
            joint_offset = joint_offsets[jI]
            fbx_skeleton_node.LclTranslation.Set(FbxDouble3(joint_offset[0], joint_offset[1], joint_offset[2]))

            fbx_skeleton_handler.joints.append(fbx_skeleton_node)
            
        for pI, child_indices in enumerate(skeleton_children):
            
            parent_node = fbx_skeleton_handler.joints[pI]
            
            for cI in child_indices:
                child_node = fbx_skeleton_handler.joints[cI]
                parent_node.AddChild(child_node)

        fbx_reference_node.AddChild(fbx_skeleton_handler.joints[0])

        fbx_skeleton_handler.root = fbx_reference_node
            
        return fbx_skeleton_handler

    @staticmethod
    def createAnimationLayer(pScene, pSkeleton, pMocap_data, pSkeletonIndex=0):
        
        skeleton_joints = pMocap_data.skeleton_joints
        rot_sequence = pMocap_data.motion_rot_sequence 
        
        motion_pos_local = pMocap_data.motion_pos_local
        motion_rot_local_euler = pMocap_data.motion_rot_local_euler

        anim_stack_name = pScene.GetName() + "_AnimStack_" + str(pSkeletonIndex)
        anim_stack = FbxAnimStack.Create(pScene, anim_stack_name)

        anim_layer_name = pScene.GetName() + "_BaseLayer_"  + str(pSkeletonIndex)
        anim_layer = FbxAnimLayer.Create(pScene, anim_layer_name)
        anim_stack.AddMember(anim_layer)

        animTime = FbxTime()

        joint_count = len(skeleton_joints)
        for jI in range(joint_count):
            
            #pos_local = motion_pos_local[:, jI, :]
            #rot_local_euler = motion_rot_local_euler[:, jI, :]
            
            pos_local = motion_pos_local[jI]
            rot_local_euler = motion_rot_local_euler[jI]

            skeleton_node = pSkeleton.joints[jI]

            node_pos_x_curve = skeleton_node.LclTranslation.GetCurve(anim_layer, "X", True)
            node_pos_y_curve = skeleton_node.LclTranslation.GetCurve(anim_layer, "Y", True)
            node_pos_z_curve = skeleton_node.LclTranslation.GetCurve(anim_layer, "Z", True)
            
            node_rot_x_curve = skeleton_node.LclRotation.GetCurve(anim_layer, "X", True)
            node_rot_y_curve = skeleton_node.LclRotation.GetCurve(anim_layer, "Y", True)
            node_rot_z_curve = skeleton_node.LclRotation.GetCurve(anim_layer, "Z", True)

            node_pos_x_curve.KeyModifyBegin()
            node_pos_y_curve.KeyModifyBegin()
            node_pos_z_curve.KeyModifyBegin()

            for fI in range(pos_local.shape[0]):
                
                if hasattr(pMocap_data, "motion_times") and jI in pMocap_data.motion_times:
                    times = pMocap_data.motion_times[jI]
                    frameTime = times[fI] if len(times) > fI else float(fI) / pMocap_data.motion_frame_rate
                else:
                    frameTime = float(fI) / pMocap_data.motion_frame_rate
                    
                animTime.SetSecondDouble(frameTime)

                # pos x
                keyIndex = node_pos_x_curve.KeyAdd(animTime)[0]
                node_pos_x_curve.KeySetValue(keyIndex, float(pos_local[fI, 0]))
                node_pos_x_curve.KeySetInterpolation(keyIndex, FbxAnimCurveDef.eInterpolationCubic)

                # pos y
                keyIndex = node_pos_y_curve.KeyAdd(animTime)[0]
                node_pos_y_curve.KeySetValue(keyIndex, float(pos_local[fI, 1]))
                node_pos_y_curve.KeySetInterpolation(keyIndex, FbxAnimCurveDef.eInterpolationCubic)

                # pos z
                keyIndex = node_pos_z_curve.KeyAdd(animTime)[0]
                node_pos_z_curve.KeySetValue(keyIndex, float(pos_local[fI, 2]))
                node_pos_z_curve.KeySetInterpolation(keyIndex, FbxAnimCurveDef.eInterpolationCubic)
            
            node_pos_x_curve.KeyModifyEnd()
            node_pos_y_curve.KeyModifyEnd()
            node_pos_z_curve.KeyModifyEnd()
            
            node_rot_x_curve.KeyModifyBegin()
            node_rot_y_curve.KeyModifyBegin()
            node_rot_z_curve.KeyModifyBegin()
            
            for fI in range(rot_local_euler.shape[0]):
                
                if hasattr(pMocap_data, "motion_times") and jI in pMocap_data.motion_times:
                    times = pMocap_data.motion_times[jI]
                    frameTime = times[fI] if len(times) > fI else float(fI) / pMocap_data.motion_frame_rate
                else:
                    frameTime = float(fI) / pMocap_data.motion_frame_rate
                    
                animTime.SetSecondDouble(frameTime)

                # rot x
                keyIndex = node_rot_x_curve.KeyAdd(animTime)[0]
                node_rot_x_curve.KeySetValue(keyIndex, float(rot_local_euler[fI, 0]))
                node_rot_x_curve.KeySetInterpolation(keyIndex, FbxAnimCurveDef.eInterpolationCubic)

                # rot y
                keyIndex = node_rot_y_curve.KeyAdd(animTime)[0]
                node_rot_y_curve.KeySetValue(keyIndex, float(rot_local_euler[fI, 1]))
                node_rot_y_curve.KeySetInterpolation(keyIndex, FbxAnimCurveDef.eInterpolationCubic)

                # rot z
                keyIndex = node_rot_z_curve.KeyAdd(animTime)[0]
                node_rot_z_curve.KeySetValue(keyIndex, float(rot_local_euler[fI, 2]))
                node_rot_z_curve.KeySetInterpolation(keyIndex, FbxAnimCurveDef.eInterpolationCubic)
            
            node_rot_x_curve.KeyModifyEnd()
            node_rot_y_curve.KeyModifyEnd()
            node_rot_z_curve.KeyModifyEnd()

        return anim_layer
    
    @staticmethod
    def exportFBX(pFilename, pSdkManager, pScene):
        
        exporter = FbxExporter.Create(pSdkManager, "")

        if not exporter.Initialize(pFilename, -1, pSdkManager.GetIOSettings()):
            print("Call to FbxExporter::Initialize() failed.")
            print("Error returned: %s" % exporter.GetStatus().GetErrorString())
            return False

        exporter.SetFileExportVersion("FBX201400")
        status = exporter.Export(pScene)
        exporter.Destroy()
        return status