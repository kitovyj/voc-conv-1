classdef vocalizations_storage < handle
   properties
      count = 0;
      path;
   end
   methods
       
      function obj = vocalizations_storage(path)
          obj.path = path;
      end          
      
      function store(obj, data)
            
        h = dialog ('visible', 'off', 'windowstyle', 'normal', 'units', 'pixels', 'position', [1 1 200 200]);
        ax = axes('parent', h);        
        
        hold on;        
        cla;
        xlim([0 0.15]);
        ylim([20000 130000]);                        

        set(ax, 'Visible', 'off') 
        
        set(ax, 'units', 'pixels', 'position', [1 1 200 200]);        
                
        p = plot(ax, data(1,:), data(2,:));
        set(p, 'LineWidth', 4);        
        
        hold off; 
        
        file_path = strcat(obj.path, sprintf('data%09d.png', obj.count));
                
        %[x, y, im, flag] = getimage(ax);                        
        frame = getframe(ax);
        im = frame.cdata;
        im = rgb2gray(im);
        im = imresize(im, [100 100]);        
                
        %saveas(ax, file_path, 'png');
        
        imwrite(im, file_path)
        
        
        %csv_file_path = strcat(file_path, '.csv');
        %csvwrite(csv_file_path, im);
                
        close(h);
        
        obj.count = obj.count + 1;        
          
      end
      
   end
end