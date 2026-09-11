"""
Load functionality adapated from:
    Autodesk C++ FBX SDK example : samples/ImportScene
Save functionality adapted from:
    Stereolabs ZED C++ SDK example : samples/export/fbx export
"""


"""
Imports
"""

from common.FbxCommon import *
from fbx import *
from fbx import FbxSkeleton

import numpy as np

# frame rate correspondences from here: https://github.com/simnalamburt/SoftwareRasterizer/blob/master/FBX/include/fbxsdk/core/base/fbxtime.h
FBX_TimeModes = {FbxTime.EMode.eDefaultMode: "default", 
             FbxTime.EMode.eFrames120: "120", 
             FbxTime.EMode.eFrames100: "100",
             FbxTime.EMode.eFrames60: "60",
             FbxTime.EMode.eFrames50: "50",
             FbxTime.EMode.eFrames48: "48",
             FbxTime.EMode.eFrames30: "30",
             FbxTime.EMode.eFrames30Drop: "30", 
             FbxTime.EMode.eNTSCDropFrame: "29.97", 
             FbxTime.EMode.eNTSCFullFrame: "29.97", 
             FbxTime.EMode.ePAL: "25",
             FbxTime.EMode.eFrames24: "24", 
             FbxTime.EMode.eFrames1000: "1000", 
             FbxTime.EMode.eFilmFullFrame: "23.976", 
             FbxTime.EMode.eCustom: "custom", 
             FbxTime.EMode.eFrames96: "96", 
             FbxTime.EMode.eFrames72: "72",
             FbxTime.EMode.eFrames59dot94: "59.94", 
             FbxTime.EMode.eFrames119dot88: "119.88"}


# skeleton handler used as intermediary data structure to save fbx
class FBX_Skeleton_Handler:
    def __init__(self):
        self.root = None
        self.joints = []

# internal representation of mocap data
class FBX_Mocap_Data:
    
    def __init__(self):
        
        self.skeleton_root_node = None
        self.skeleton_root = ""
        self.skeleton_nodes = []
        self.skeleton_joints = []
        self.skeleton_children = []
        self.skeleton_parents = []
        self.skeleton_joint_offsets = []
        
        self.motion_rot_sequence = []
        self.motion_frame_rate = -1
        self.motion_frame_count = 0
        self.motion_pos_local = None
        self.motion_rot_local_euler = None
        
        self.system_unit = None

class FBX_Tools():
    
    def __init__(self, filename=None):
        self.reset()
        
    def reset(self): 
        self.mocap_data = []

    # TODO: make this work with multiple animation layers
    def load(self, mocap_file_path):
        self.reset()
        
        # Prepare FBX SDK
        self.sdkManager, self.scene = InitializeSdkObjects()

        # Load Scene
        result = LoadScene(self.sdkManager, self.scene, mocap_file_path)
        
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
            self.animation_layers += FBX_Tools.getAnimationLayers(anim_stack)
            
        if len(self.animation_layers) == 0:
            print("no animation layers found")
            return
        elif len(self.animation_layers) > 1:
            print("Warning: more than one animation layer found but only one layer supported at the moment")
            return
            
        # at the moment, this works only with first animation layer
        # TODO: figure out which animation layers are really relevant for skeletons
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
            skel_mocap_data.skeleton_children = skeleton_data["children"]
            skel_mocap_data.skeleton_parents = skeleton_data["parents"]
            skel_mocap_data.skeleton_joint_offsets = skeleton_data["offsets"]
            

        # collection motion data for each mocap data
        for skel_mocap_data in self.mocap_data:
            
            # get frame rate
            skel_mocap_data.motion_frame_rate = FBX_Tools.getFrameRate(self.scene)
            
            # get systemm unit
            su = self.scene.GetGlobalSettings().GetSystemUnit()
            
            # get frame count
            skel_mocap_data.motion_frame_count = FBX_Tools.getFrameCount(root, self.animation_layer)
            
            # assumes, that all nodes have the same rotatioin sequence as the skeleton root node
            skel_mocap_data.motion_rot_sequence = FBX_Tools.getRotationSequence(skel_mocap_data.skeleton_root_node )
            
            pos_local, rot_local_euler = FBX_Tools.getMotion(skel_mocap_data.skeleton_nodes, self.animation_layer, skel_mocap_data.motion_frame_count )
            
            skel_mocap_data.motion_pos_local = pos_local
            skel_mocap_data.motion_rot_local_euler = rot_local_euler
            
        return self.mocap_data

    # TODO: test (and possibly fix) for multiple skeletons
    # assumes that all skeletons have identical topology
    def write(self, mocap_data, fileName):
        
        self.reset()
        self.mocap_data = mocap_data
        
        # set framerate for scene
        FBX_Tools.setFrameRate(self.scene, mocap_data)
        
        # set system unit for scene
        FBX_Tools.setSystemUnit(self.scene, mocap_data)

        # Prepare the FBX SDK
        self.sdkManager, self.scene = InitializeSdkObjects()
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
    # TODO: verify that this really works for any FBX file
    @staticmethod
    def getSkeletonRootNodes(pScene):
        
        skeletonRoots = []
        
        sceneRoot = pScene.GetRootNode()

        if sceneRoot:
            
            FBX_Tools.findSkeletonRootNode(sceneRoot, skeletonRoots)
            
            """
            # not in every fbx file are the first children of the sceneroot skeleton roots
            # therefore, this doesn't always work
            for i in range(sceneRoot.GetChildCount()):
                
                childNode = sceneRoot.GetChild(i)

                # for some reason, in the zed fbx export, the nodes don't have an attribute'
                childAttribute = childNode.GetNodeAttribute()

                if childAttribute is None:
                    continue
                
                childAttributeType = (childAttribute.GetAttributeType())
                
                if childAttributeType is not FbxNodeAttribute.EType.eSkeleton:
                    continue
                
                skeletonRoots.append(childNode)
            """

        return skeletonRoots
    
    # traverse node hiearchy until first skeleton nodes are found
    @staticmethod
    def findSkeletonRootNode(pNode, skelRootNodes):
        
        #print("findSkeletonRootNode")
        
        #print("pNode name ", pNode.GetName())
        
        nodeAttribute = pNode.GetNodeAttribute()
        if nodeAttribute is not None:
            nodeAttributeType = (nodeAttribute.GetAttributeType())
            
            #print("pNode type ", nodeAttributeType)
            
            if nodeAttributeType is FbxNodeAttribute.EType.eSkeleton:
                
                #print("is skeleton root node")
                
                skelRootNodes.append(pNode)
                return
        
        #print("is not skeleton root node, continue search")
        
        childCount = pNode.GetChildCount()
        for i in range(pNode.GetChildCount()):
            childNode = pNode.GetChild(i)
            FBX_Tools.findSkeletonRootNode(childNode, skelRootNodes)
        
    
    # TODO: figure out how to get custom framerate
    @staticmethod
    def getFrameRate(pScene):
        
        timeMode = pScene.GetGlobalSettings().GetTimeMode()
        fps_string = FBX_TimeModes[timeMode]
        
        try:
            fps_float = float(fps_string)
        except ValueError:
            fps_float = -1
            
        return fps_float

    # TODO: figure out how to get custom framerate
    @staticmethod
    def setFrameRate(pScene, mocap_data):
        
        fps = mocap_data[0].motion_frame_rate
        
        if fps.is_integer() == True:
            fps = int(fps)
        
        fps_string = "{}".format(fps)
        
        print("fps ", fps)
        print("fps_string ", fps_string)
        
        timeMode = FbxTime.EMode.eDefaultMode
        
        for key, value in FBX_TimeModes.items():
            
            print("value ", value)
            
            
            if value == fps_string:
                timeMode = key
                
                print("set time mode to : ", value)
                
                break
        
        pScene.GetGlobalSettings().SetTimeMode(timeMode)
        
    @staticmethod
    def getSystemUnit(pScene):
        
        su = pScene.GetGlobalSettings().GetSystemUnit()
        
        if su == FbxSystemUnit.cm:
            return "cm"
        if su == FbxSystemUnit.dm:
            return "dm"
        if su == FbxSystemUnit.Foot:
            return "foot"
        if su == FbxSystemUnit.Inch:
            return "inch"            
        if su == FbxSystemUnit.km:
            return "km"             
        if su == FbxSystemUnit.m:
            return "m"    
        if su == FbxSystemUnit.mm:
            return "mm"  
        if su == FbxSystemUnit.Yard:
            return "yard"  

        return "unknown"


    @staticmethod
    def setSystemUnit(pScene, mocap_data):
        
        su_string = mocap_data[0].system_unit
        
        if su_string == "cm":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.cm) 
            return
        if su_string == "dm":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.dm) 
            return
        if su_string == "foot":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.Foot) 
            return
        if su_string == "inch":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.Inch) 
            return        
        if su_string == "km":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.km) 
            return    
        if su_string == "m":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.m) 
            return    
        if su_string == "mm":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.mm) 
            return                
        if su_string == "yard":
            pScene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.Yard) 
            return   

    # get animation stacks
    @staticmethod
    def getAnimationStacks(pScene):
        
        animationStacks = []
        
        for i in range(pScene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))):
            lAnimStack = pScene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
            
            animationStacks.append(lAnimStack)

        return animationStacks
    
    # get animation layers
    @staticmethod
    def getAnimationLayers(pAnimStack):
        
        animationLayers = []
        
        animationLayerCount = pAnimStack.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimLayer.ClassId))
        
        for l in range(animationLayerCount):

            animationLayer = pAnimStack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), l)
            animationLayers.append(animationLayer)

        return animationLayers
    
    # get frame count
    # TODO: check if it is a reliable method to count the number of posX values for the skeleton root node
    @staticmethod
    def getFrameCount(pSkeletonRootNode, pAnimLayer):

        posXCurve = pSkeletonRootNode.LclTranslation.GetCurve(pAnimLayer, "X")
        posXValues = FBX_Tools.getNodeValues(pSkeletonRootNode, posXCurve)
        
        return len(posXValues)
    
    # get rotation sequence
    @staticmethod
    def getRotationSequence(pNode):
        lRotationOrder = pNode.GetRotationOrder(FbxNode.EPivotSet.eSourcePivot)
        if lRotationOrder == EFbxRotationOrder.eEulerXYZ:
            return [0, 1, 2]
        elif lRotationOrder == EFbxRotationOrder.eEulerXZY:
            return [0, 2, 1]
        elif lRotationOrder == EFbxRotationOrder.eEulerYZX:
            return [1, 2, 0]
        elif lRotationOrder == EFbxRotationOrder.eEulerYXZ:
            return [1, 0, 2]
        elif lRotationOrder == EFbxRotationOrder.eEulerZXY:
            return [2, 0, 1]
        elif lRotationOrder == EFbxRotationOrder.eEulerZYX:
            return [2, 1, 0]
        elif lRotationOrder == EFbxRotationOrder.eSphericXYZ:
            # TODO: check how this is different from eEulerXYZ
            return [0, 1, 2]
        
    # set rotation sequence
    @staticmethod
    def setRotationSequence(pNode, pRotSequence):
        
        # test
        pNode.SetRotationActive(True)
        
        if pRotSequence == [0, 1, 2]:     
            pNode.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, EFbxRotationOrder.eEulerXYZ)
        elif pRotSequence == [0, 2, 1]:
            pNode.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, EFbxRotationOrder.eEulerXZY)
        elif pRotSequence == [1, 2, 0]:
            pNode.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, EFbxRotationOrder.eEulerYZX)
        elif pRotSequence == [1, 0, 2]:
            pNode.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, EFbxRotationOrder.eEulerYXZ)
        elif pRotSequence == [2, 0, 1]:
            pNode.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, EFbxRotationOrder.eEulerZXY)
        elif pRotSequence == [2, 1, 0]:
            pNode.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, EFbxRotationOrder.eEulerZYX)
    
    # get all nodes that belong to a single skeleton
    @staticmethod
    def getSkeletonNodes(pSkeletonRootNode):
        
        nodes = []
        FBX_Tools.traverseSkeletonNodes(pSkeletonRootNode, nodes)
        
        return nodes
    
    # recursively gather all nodes
    @staticmethod
    def traverseSkeletonNodes(pNode, pNodes):
        
        # # debug begin
        # nodeAttribute = pNode.GetNodeAttribute()
        # if nodeAttribute is not None:
        #     nodeAttributeType = (nodeAttribute.GetAttributeType())
        #     print("pNode name ", pNode.GetName())
        #     print("pNode type ", nodeAttributeType)
        # # debug end
        
        pNodes.append(pNode)
        
        childCount = pNode.GetChildCount()
        for i in range(childCount):
            childNode = pNode.GetChild(i)
            
            FBX_Tools.traverseSkeletonNodes(childNode, pNodes)
            
    # get node values from a single curve
    @staticmethod
    def getNodeValues(pNode, pCurve):
        
        keyValues = []
        
        if pCurve is not None:

            #interpolation = [ "?", "constant", "linear", "cubic"]
            #constantMode =  [ "?", "Standard", "Next" ]
            #cubicMode =     [ "?", "Auto", "Auto break", "Tcb", "User", "Break", "User break" ]
            #tangentWVMode = [ "?", "None", "Right", "Next left" ]
        
            keyCount = pCurve.KeyGetCount()
        
            for key in range(keyCount):
                lTimeString = ""
                lKeyValue = pCurve.KeyGetValue(key)
                #lKeyTime  = pCurve.KeyGetTime(key)
                #lInterpolation = pCurve.KeyGetInterpolation(key)
                
                #print("key ", key)
                #print("lKeyValue t ", type(lKeyValue))
                #print("lKeyTime t ", type(lKeyTime))
                #print("lInterpolation t ", type(lInterpolation))
                
                keyValues.append(lKeyValue)
            
        return keyValues
            
    # get local position and local rotation (euler) of a node
    @staticmethod
    def getNodeRotPos(pNode, pAnimLayer, pFrameCount):
        
        #print("getNodeRotPos node ", pNode.GetName())

        posXCurve = pNode.LclTranslation.GetCurve(pAnimLayer, "X")
        posYCurve = pNode.LclTranslation.GetCurve(pAnimLayer, "Y")
        posZCurve = pNode.LclTranslation.GetCurve(pAnimLayer, "Z")
        
        rotXCurve = pNode.LclRotation.GetCurve(pAnimLayer, "X")
        rotYCurve = pNode.LclRotation.GetCurve(pAnimLayer, "Y")
        rotZCurve = pNode.LclRotation.GetCurve(pAnimLayer, "Z")
        
        posXValues = FBX_Tools.getNodeValues(pNode, posXCurve)
        posYValues = FBX_Tools.getNodeValues(pNode, posYCurve)
        posZValues = FBX_Tools.getNodeValues(pNode, posZCurve)
        
        #print("node ", pNode.GetName())
        #print("len posXValues ", len(posXValues))
        #print("len posYValues ", len(posXValues))
        #print("len posZValues ", len(posXValues))
        
        # if they position values are empty, replace them with offset values
        # if they are non-terminal nodes, otherwise replace them with zeros
        # TODO: check if this is correct
        offset = pNode.LclTranslation.Get()
        childCount = pNode.GetChildCount()
        
        if len(posXValues) == 0:
            if childCount > 0:
                posXValues = [offset[0]] * pFrameCount
            else:
                posXValues = [0] * pFrameCount
        if len(posYValues) == 0:
            if childCount > 0:
                posYValues = [offset[1]] * pFrameCount
            else:
                posYValues = [0] * pFrameCount
        if len(posZValues) == 0:
            if childCount > 0:
                posZValues = [offset[2]] * pFrameCount
            else:
                posZValues = [0] * pFrameCount

        rotXValues = FBX_Tools.getNodeValues(pNode, rotXCurve)
        rotYValues = FBX_Tools.getNodeValues(pNode, rotYCurve)
        rotZValues = FBX_Tools.getNodeValues(pNode, rotZCurve)
        
        #print("len rotXValues ", len(rotXValues))
        #print("len rotYValues ", len(rotYValues))
        #print("len rotZValues ", len(rotZValues))
        
        # if they rotation values are empty, replace them with zero values
        if len(rotXValues) == 0:
            rotXValues = [0] * pFrameCount
        if len(rotYValues) == 0:
            rotYValues = [0] * pFrameCount
        if len(rotZValues) == 0:
            rotZValues = [0] * pFrameCount

        posValues = []
        
        posValues.append(posXValues)
        posValues.append(posYValues)
        posValues.append(posZValues)
        
        rotValues = []

        rotValues.append(rotXValues)
        rotValues.append(rotYValues)
        rotValues.append(rotZValues)    
        
        return posValues, rotValues
              

    # get motion data (pos local and rot local euler)
    def getMotion(pNodes, pAnimLayer, pFrameCount):
        
        #print("getMotion")

        pos_local = []
        rot_local_euler = []    

        node_posrot_dict = {}
        
        for node in pNodes:
            
            #print("node ", node.GetName())
            
            node_pos, node_rot_euler = FBX_Tools.getNodeRotPos(node, pAnimLayer, pFrameCount)

            node_pos = np.transpose(np.array(node_pos))
            node_rot_euler = np.transpose(np.array(node_rot_euler))
            
            #print("node_pos s ", node_pos.shape)
            #print("node_rot_euler s ", node_rot_euler.shape)
            
            pos_local.append(node_pos)
            rot_local_euler.append(node_rot_euler)
            
        pos_local = np.stack(pos_local, axis=1)
        rot_local_euler = np.stack(rot_local_euler, axis=1)
        
        return pos_local, rot_local_euler
    
    # recursively construct skeleton topology
    @staticmethod
    def traverseSkeletonNodeHierarchy(pParentNode, pNode, skeletonData):
        
        #print("traverseSkeletonNodeHierarchy begin")
        
        # get joint parent
        if pParentNode is None:
            parentName = ""
        else:
            parentName = pParentNode.GetName()
        skeletonData["parents"].append(parentName)
        #print("parent ", parentName)
        
        # get joint name
        jointName = pNode.GetName()
        skeletonData["joints"].append(jointName)
        
        #print("jointName ", jointName)
        #print("skeleton joints ", skeleton["joints"])
        
        # get node offset
        tr = pNode.LclTranslation.Get()
        offset = [tr[0], tr[1], tr[2]]
        skeletonData["offsets"].append(offset)
        #print("offset ", offset)
        #print("skeleton offsets ", skeleton["offsets"])
        
        # get node children
        children = []
        childCount = pNode.GetChildCount()
        for i in range(childCount):
            childNode = pNode.GetChild(i)
            childName = childNode.GetName()
            children.append(childName)
        skeletonData["children"].append(children)
        #print("children ", children)
        #print("skeleton children ", skeleton["children"])
        
        #print("traverseSkeletonNodeHierarchy end")
        
        # traverse node hierarchy
        for i in range(childCount):
            childNode = pNode.GetChild(i)
            FBX_Tools.traverseSkeletonNodeHierarchy(pNode, childNode, skeletonData)
        
    # get skeleton daza
    # TODO: verify that the first skeleton node is indeed the skeletons root node
    @staticmethod
    def getSkeletonData(pSkeletonRootNode):
        
        #print("getSkeleton begin")

        skeletonData = {
                    "root": "",
                    "joints": [],
                    "offsets": [],
                    "parents": [],
                    "children": []
                    }

        skeletonData["root"] = pSkeletonRootNode.GetName()
        #print("root: ", skeleton["root"])
        
        FBX_Tools.traverseSkeletonNodeHierarchy(None, pSkeletonRootNode, skeletonData)
        
        # convert offsets into numpy array
        skeletonData["offsets"] = np.array(skeletonData["offsets"])
        
        # convert parent and children names to indices
        parent_indices = []
        for parent_name in skeletonData["parents"]:
            if parent_name in skeletonData["joints"]:
                parent_indices.append( skeletonData["joints"].index(parent_name) )
            else:
                parent_indices.append(-1)
        
        children_indices = []
        for children_names in skeletonData["children"]:
            joint_children_indices = []
            for child_name in children_names:
                joint_children_indices.append( skeletonData["joints"].index(child_name) )
            children_indices.append(joint_children_indices)

        skeletonData["parents"] = parent_indices
        skeletonData["children"] = children_indices
        
        #print("getSkeleton end")
        
        return skeletonData
    
    # create a skeleton representation used for saving fbx
    @staticmethod
    def createSkeleton(pScene, pMocap_data):
        
        #print("CreateSkeleton begin")
        
        fbx_skeleton_handler = FBX_Skeleton_Handler()
        
        # get mocap info 
        skeleton_joints = pMocap_data.skeleton_joints
        skeleton_children = pMocap_data.skeleton_children
        joint_offsets = pMocap_data.skeleton_joint_offsets 
        rot_sequence = pMocap_data.motion_rot_sequence 
        
        #motion_pos_local = motion_data["pos_local"]
        #motion_rot_local_euler = motion_data["rot_local_euler"]
        
        # TODO: check if I can avoid creating this reference node
        fbx_reference_node = FbxNode.Create(pScene, ("Skeleton"))
        FBX_Tools.setRotationSequence(fbx_reference_node, rot_sequence)

        # Create all skeleton nodes
        #print("Create all skeleton nodes")
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
            tr = FbxDouble3(joint_offset[0], joint_offset[1], joint_offset[2])
            fbx_skeleton_node.LclTranslation.Set(tr)
            
            fbx_skeleton_handler.joints.append(fbx_skeleton_node)
            
            # add node to mocap_data
            if jI == 0:
                pMocap_data.skeleton_root_node = fbx_skeleton_node
            pMocap_data.skeleton_nodes.append(fbx_skeleton_node)
            
        fbx_reference_node.AddChild(fbx_skeleton_handler.joints[0])
        
        # Build skeleton node hierarchy
        #print("Build skeleton node hierarchy")
        for jI in range(len(fbx_skeleton_handler.joints)):
            
            #print("parent joint jI ", jI, " name ", fbx_skeleton_handler.joints[jI].GetName(), " child_count ", len(skeleton_children[jI]))
            
            for cI in range(len(skeleton_children[jI])):
                
                #print("child joint cI ", cI, " name ", fbx_skeleton_handler.joints[skeleton_children[jI][cI]].GetName())
                
                fbx_skeleton_handler.joints[jI].AddChild(fbx_skeleton_handler.joints[skeleton_children[jI][cI]])
            
        fbx_skeleton_handler.root = fbx_reference_node
        
        return fbx_skeleton_handler

    @staticmethod
    def createAnimationLayer(pScene, pSkeleton, pMocap_data, pIndex):
        
        vec_str_component = ["X", "Y", "Z"]
        
        fbx_time = FbxTime()
        # here should a proper timestamp be calculated that represents when a skeleton appeared in the scene
        fbx_time.SetMilliSeconds(0)
        
        fbx_anim_stack = FbxAnimStack.Create(pScene, "Anim Stack ID " + str(pIndex))
        # Create the base layer (this is mandatory)   
        fbx_anim_base_layer = FbxAnimLayer.Create(pScene, "Base Layer " + str(pIndex))
        fbx_anim_stack.AddMember(fbx_anim_base_layer)

        fbx_anim_id = fbx_anim_base_layer
        
        mocap_frame_count = pMocap_data.motion_frame_count
        mocap_frame_rate = pMocap_data.motion_frame_rate
        mocap_skeleton_joints = pMocap_data.skeleton_joints
        mocap_motion_rot_local_euler = pMocap_data.motion_rot_local_euler 
        mocap_motion_pos_local = pMocap_data.motion_pos_local
        
        """
        mocap_skeleton = pMocap_data["skeleton"]
        mocap_motion = pMocap_data["motion"]
        mocap_motion_pos_local = pMocap_data["pos_local"]
        mocap_motion_rot_local_euler = pMocap_data["rot_local_euler"]
        
        mocap_frame_count = mocap_motion_pos_local.shape[0]
        mocap_frame_rate = pMocap_data["frame_rate"]
        mocap_joints = mocap_skeleton["joints"]
        """
        
        # For each frame (keypoint) 
        for fI in range(mocap_frame_count):
            #print("fI ", fI)
            
            # frame time in milisecs
            frame_time = int(fI / mocap_frame_rate * 1000)
            fbx_time.SetMilliSeconds(frame_time)
            
            #print("frame_time ", frame_time)
        
            # for each joint
            for jI in range(len(mocap_skeleton_joints)):
                
                #print("jI ", jI)
                
                fbx_joint = pSkeleton.joints[jI]
                mocap_joint_rot_local_euler = mocap_motion_rot_local_euler[fI, jI]
                
                #print("mocap_joint_rot_local_euler ", mocap_joint_rot_local_euler)
                
                # Set translation of the root (first joint)
                if jI == 0: 
                    mocap_root_position = mocap_motion_pos_local[fI, jI]
                    fbx_joint.LclTranslation.GetCurveNode(fbx_anim_id, True)
                    
                    for d in range(3):
                        fbx_lCurve = fbx_joint.LclTranslation.GetCurve(fbx_anim_id, vec_str_component[d], True)
                        if fbx_lCurve:
                            fbx_lCurve.KeyModifyBegin()
                            fbx_key_index = fbx_lCurve.KeyAdd(fbx_time) # TODO: find out why this returns a tuple which is incompatible with fbx_lCurve.KeySet
                            fbx_key_index = fI
                            fbx_lCurve.KeySet(fbx_key_index, fbx_time, mocap_root_position[d], FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
                            fbx_lCurve.KeyModifyEnd()
                    
                    #Use global rotation for the root
                    #not necessary here since the rotation stored for the root joint is already in global coordinates
                    #fbx_lcl_rotation = mocap_motion_rot_local_euler[fI, jI]
                
                #Convert rotation to euler angles
                # not necessary since orientation is already in euler angles
                # FbxQuaternion quat = FbxQuaternion(lcl_rotation.x, lcl_rotation.y, lcl_rotation.z, lcl_rotation.w);
                # FbxVector4 rota_euler;
                # rota_euler.SetXYZ(quat);
                
                #Set local rotation of the joint
                mocap_joint_rotation_euler = mocap_motion_rot_local_euler[fI, jI]
                for d in range(3):
                    fbx_lCurve = fbx_joint.LclRotation.GetCurve(fbx_anim_id, vec_str_component[d], True)
                    if fbx_lCurve:
                        fbx_lCurve.KeyModifyBegin()
                        fbx_key_index = fbx_lCurve.KeyAdd(fbx_time) # TODO: find out why this returns a tuple which is incompatible with fbx_lCurve.KeySet 
                        fbx_key_index = fI
                        fbx_lCurve.KeySet(fbx_key_index, fbx_time, mocap_joint_rotation_euler[d], FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
                        fbx_lCurve.KeyModifyEnd()
        
        return fbx_anim_base_layer

    @staticmethod
    def exportFBX(fileName, pFbxManager, pScene):
        
        fbx_lStatus = True
        
        # Create an exporter.
        fbx_lExporter = FbxExporter.Create(pFbxManager, "")
        fbx_fileFormat = 0 # save as binary
        fbx_embedMedia = False
        
        if fbx_fileFormat < 0 or fbx_fileFormat >= pFbxManager.GetIOPluginRegistry().GetWriterFormatCount():
            # Write in fall back format in less no ASCII format found
            fbx_fileFormat = pFbxManager.GetIOPluginRegistry().GetNativeWriterFormat()
            # Try to export in ASCII if possible
            fbx_lFormatCount = pFbxManager.GetIOPluginRegistry().GetWriterFormatCount()
            
            for fbx_lFormatIndex in range(fbx_lFormatCount):
                if pFbxManager.GetIOPluginRegistry().WriterIsFBX(fbx_lFormatIndex):
                    fbx_lDesc = pFbxManager.GetIOPluginRegistry().GetWriterFormatDescription(fbx_lFormatIndex)
        
                    if fbx_lDesc.Find("ascii")>=0:
                        fbx_fileFormat = fbx_lFormatIndex
                        break
        
        # Set the export states. By default, the export states are always set to 
        # true except for the option eEXPORT_TEXTURE_AS_EMBEDDED. The code below 
        # shows how to change these states.
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_MATERIAL, True)
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_TEXTURE, True)
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, fbx_embedMedia)
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_SHAPE, True)
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_GOBO, True)
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_ANIMATION, True)
        pFbxManager.GetIOSettings().SetBoolProp(EXP_FBX_GLOBAL_SETTINGS, True)
        
        # Initialize the exporter by providing a filename.
        if fbx_lExporter.Initialize(fileName, fbx_fileFormat, pFbxManager.GetIOSettings()) == False :
            print("Call to FbxExporter::Initialize() failed.\n")
            print("Error returned: {}\n\n".format(fbx_lExporter.GetStatus().GetErrorString()))
            assert(False)

        lMajor, lMinor, lRevision = FbxManager.GetFileFormatVersion()
        print("FBX file format version {}.{}.{}\n\n".format(lMajor, lMinor, lRevision))
        
        # Export the scene.
        fbx_lStatus = fbx_lExporter.Export(pScene)
        
        # Destroy the exporter.
        fbx_lExporter.Destroy()
      
      
