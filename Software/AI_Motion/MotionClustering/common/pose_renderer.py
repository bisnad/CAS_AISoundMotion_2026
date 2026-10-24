import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


class PoseRenderer:
    def __init__(self, edge_data):
        self.edge_data = edge_data
    
    def _fig2data (self, fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )
 
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        
        #print("w ", w, " h ", h)
        
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h,4 )
 
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        return buf
    
    def _fig2img (self, fig):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image
        """
        # put the figure pixmap into a numpy array
        buf = self._fig2data ( fig )
        w, h, d = buf.shape
        
        return Image.frombuffer( "RGBA", ( w ,h ), buf.tostring( ) )
    
    def create_pose_image(self, pose, axis_min, axis_max, rot_elev, rot_azi, line_width, image_xinch, image_yinch):
        point_data = np.array([pose[:,0], pose[:,1], pose[:,2]])
        lines_data = np.array([[pose[edge[0],:], pose[edge[1],:]] for edge in self.edge_data])

        fig = plt.figure(figsize=(image_xinch,image_yinch)) 
        plt.axis("off")
        fig.tight_layout()
            
        ax = plt.axes(projection="3d")
        ax.view_init(elev=rot_elev, azim=rot_azi)
        
        ax.set_xlim(axis_min[0], axis_max[0])
        ax.set_ylim(axis_min[1], axis_max[1])
        ax.set_zlim(axis_min[2], axis_max[2])
            
        # Make panes transparent
        ax.xaxis.pane.fill = False # Left pane
        ax.yaxis.pane.fill = False # Right pane
        ax.zaxis.pane.fill = False # Right pane
            
        ax.grid(False) # Remove grid lines
            
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
            
        # Transparent spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            
        # Transparent panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
            
        for line in lines_data:
            ax.plot(line[:,0], line[:,1], zs=line[:,2], linewidth=line_width, color='cadetblue', alpha=0.5)
            ax.scatter(point_data[0, :], point_data[1, :], point_data[2, :], s=line_width * 8, color='darkslateblue', alpha=0.5)
    
        fig.show()
    
        pose_image = self._fig2img ( fig )
    
        plt.close()
        
        return pose_image
    
    def create_pose_images(self, poses, axis_min, axis_max, rot_elev, rot_azi, line_width, image_xinch, image_yinch):
        pose_count = poses.shape[0]
        pose_images = []
        
        fig = plt.figure(figsize=(image_xinch,image_yinch)) 
        plt.axis("off")
        fig.tight_layout()
        
        ax = plt.axes(projection="3d")
        ax.view_init(elev=rot_elev, azim=rot_azi)
        
        ax.set_xlim(axis_min[0], axis_max[0])
        ax.set_ylim(axis_min[1], axis_max[1])
        ax.set_zlim(axis_min[2], axis_max[2])
        
        # Make panes transparent
        ax.xaxis.pane.fill = False # Left pane
        ax.yaxis.pane.fill = False # Right pane
        ax.zaxis.pane.fill = False # Right pane
        
        ax.grid(False) # Remove grid lines
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Transparent spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        scatter_data = None
        
        fig.show()
        
        for pI in range(pose_count):
            
            # cleanup previous drawing
            if scatter_data != None:
                scatter_data.remove()
            if len(ax.lines) > 0:
                ax.lines.clear()
            
            point_data = np.array([poses[pI, :,0], poses[pI, :,1], poses[pI,:,2]])
            lines_data = np.array([[poses[pI, edge[0],:], poses[pI, edge[1],:]] for edge in self.edge_data])
            
            for line in lines_data:
                ax.plot(line[:,0], line[:,1], zs=line[:,2], linewidth=line_width, color='cadetblue', alpha=0.5)
            scatter_data = ax.scatter(point_data[0, :], point_data[1, :], point_data[2, :], s=line_width*8.0, color='darkslateblue', alpha=0.5)

            im = self._fig2img ( fig )
            
            pose_images.append(im)
    
        plt.close()
            
        return pose_images
 
    def create_grid_image(self, images, grid):
        h_count = grid[0]
        v_count = grid[1]

        fig = plt.figure(figsize=(h_count * 2, v_count * 2))

        image_count = h_count * v_count

        for iI in range(image_count):
            ax = fig.add_subplot(v_count, h_count, iI + 1)
            ax.imshow(images[iI])
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()

        fig.show()
    
        grid_image = self._fig2img ( fig )
    
        plt.close()
        
        return grid_image
    