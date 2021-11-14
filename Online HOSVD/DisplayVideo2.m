function  DisplayVideo2( Data1,Data2,Data3,Data4,VideoName )


writerObj = VideoWriter(VideoName); % Name it.
writerObj.FrameRate = 15; % How many frames per second.
open(writerObj); 
figure;
n = size(Data1,3);
for i = 1: n
    i
   img1 = Data1(:,:,i);

   
   img2 = Data2(:,:,i);
 
    
   img3 = Data3(:,:,i);

   
   img4 = Data4(:,:,i);
  
 h = subplot('position',[0.01,0.50,0.47,0.42]);
    imshow(((img1)));
     title('original')
      
    h = subplot('position',[0.49,0.50,0.47, 0.42]);
    imshow(((img2)));
    title('noisy')
    
    h = subplot('position',[0.01,0.001,0.47, 0.42]);
    imshow(((img3)));
    title('offline')

    
    h = subplot('position',[0.49,0.001,0.47, 0.42]);
    imshow(((img4)));
     title('online')
   
      
       frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
        writeVideo(writerObj, frame);
   
    
end

hold off
close(writerObj);

