classdef vocalizations_storage < handle
   properties
      count = 0;
      path;
      h;
      ax;
   end
   methods
       
      function obj = vocalizations_storage(path, start_count)
          
          obj.path = path;
          obj.count = start_count;

          obj.h = dialog ('visible', 'off', 'windowstyle', 'normal', 'units', 'pixels', 'position', [1 1 200 200]);
          obj.ax = axes('parent', obj.h);        

          cla(obj.ax);
                      
          set(obj.ax, 'Visible', 'off')         
          set(obj.ax, 'units', 'pixels', 'position', [1 1 200 200]);
          xlim(obj.ax, [0 0.17]);
          ylim(obj.ax, [30000 100000]);            
          hold(obj.ax, 'on');
          
      end          
      
      function delete(obj)
                  
        close(obj.h);          
        
      end      
      
      function store(obj, data, labels, break_num)
                                   
        cla(obj.ax);
        
        breaks = generate_breaks(0, 1000, break_num);
        
        sp = 1;

        break_size = 20;
        
        for i=1:numel(breaks)
            np = int32(breaks(i));
            plot(obj.ax, data(1,sp:np), data(2,sp:np), 'Linewidth', 2.1);
            sp = np + break_size;            
        end
        plot(obj.ax, data(1,sp:end), data(2,sp:end), 'Linewidth', 2.1);

        %{
        if with_a_break            
            break_pos = 100 + int32(rand()*600);          
            plot(obj.ax, data(1,1:break_pos), data(2,1:break_pos), 'LineWidth', 2);
            plot(obj.ax, data(1,(break_pos + 20):end), data(2,(break_pos + 20):end), 'LineWidth', 2);
        else
            plot(obj.ax, data(1,:), data(2,:), 'LineWidth', 2);            
        end
        %}
                  
        file_name = strcat(obj.path, sprintf('data%09d', obj.count)); 
        png_file = strcat(file_name, '.png');
        csv_file = strcat(file_name, '.csv');
                
        %[x, y, im, flag] = getimage(ax);                        
        frame = getframe(obj.ax);
        im = frame.cdata;
        im = rgb2gray(im);
        im = imnoise(im, 'salt & pepper');
        im = imresize(im, [100 100]);        
                
        %saveas(ax, file_path, 'png');
      
        imwrite(im, png_file)
        
        csvwrite(csv_file, labels);
                        
        obj.count = obj.count + 1;   
        
        clear im;
          
      end
      
   end
end