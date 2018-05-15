%method = 'Linear';
method = 'Ridge';
%method = 'SVM';

shuffled = 1;

load('statistics_unbalanced_new.mat');
    
[female, male, overall] = analyze_raw_input(raw_100, responses, 'Method', method, 'Shuffled', shuffled);

fn = [method '_Full'];

if shuffled
    fn = [fn '_Shuffled.csv'];
else
    fn = [fn '_Real.csv'];
end

id = fopen(fn, 'w');
fprintf(id,'"Female", "Male", "Overall"\n');

for i = 1:length(female)

    f = female(i);
    m = male(i);
    o = overall(i);
        
    fprintf(id,'%f, %f, %f\n', f, m, o);
       
end

fclose(id);    
