classdef vocalizations_storage < handle
   properties
      count = 0;
      path;
      h;
      ax;
   end
   methods
       
      function obj = vocalizations_storage(path)
          
          obj.path = path;

          obj.h = dialog ('visible', 'off', 'windowstyle', 'normal', 'units', 'pixels', 'position', [1 1 200 200]);
          obj.ax = axes('parent', obj.h);        

          cla;
          xlim(obj.ax, [0 0.15]);
          ylim(obj.ax, [20000 130000]);                        
          set(obj.ax, 'Visible', 'off')         
          set(obj.ax, 'units', 'pixels', 'position', [1 1 200 200]);        
          
      end          
      
      function delete(obj)
                  
        close(obj.h);          
        
      end      
      
      function store(obj, data)
                                   
        p = plot(obj.ax, data(1,:), data(2,:), 'LineWidth', 3);
        %set(p, 'LineWidth', 4);        
                
        file_path = strcat(obj.path, sprintf('data%09d.png', obj.count));
                
        %[x, y, im, flag] = getimage(ax);                        
        frame = getframe(obj.ax);
        im = frame.cdata;
        im = rgb2gray(im);
        im = imnoise(im, 'salt & pepper');
        im = imresize(im, [100 100]);        
                
        %saveas(ax, file_path, 'png');
        
        
        imwrite(im, file_path)
        
        
        %csv_file_path = strcat(file_path, '.csv');
        %csvwrite(csv_file_path, im);
                        
        obj.count = obj.count + 1;        
          
      end
      
   end
end