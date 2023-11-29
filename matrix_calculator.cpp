#include <bits/stdc++.h>

using namespace std;
    ostream& operator<<(ostream& s, vector<vector<double>> mat){
        s << ">>";
        for(int i = 0; i < mat.size(); i++){
            if(i != 0) s << "  ";
            for(int j = 0; j < mat[i].size(); j++){ 
                s << mat[i][j] << ' ';
            }
            s << '\n';
        }
        return s;
    }


namespace matcalc{
    const double epsilon = 1e-8;

    void correct_false_0(std::vector<std::vector<double>>& matrix){
        if(matrix.size() == 0 || matrix[0].size() == 0) return;

        for(int i = 0; i < matrix.size(); i++)
            for(int j = 0; j < matrix[0].size(); j++)
                if(matrix[i][j] > -epsilon && matrix[i][j] < epsilon) matrix[i][j] = 0;
    }

    void print_matrix(std::vector<std::vector<double>> matrix){       //OK
        if(matrix.empty()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input. Can't print empty matrix." << '\n';
            #endif
            return;
        }
        correct_false_0(matrix);
        std::cout << ">>";
        for(int i = 0; i < matrix.size(); i++){
            if(i != 0) std::cout << "  ";
            for(int j = 0; j < matrix[i].size(); j++){ 
                std::cout << matrix[i][j] << ' ';
            }
            std::cout << '\n';
        }
    }

    double dot(const std::vector<double>& vec1, const std::vector<double>& vec2){       //OK
        if(vec1.empty() || vec2.empty() || vec1.size() != vec2.size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In dot)" << '\n';
            #endif
            return 0;
        }
        double ret {};
        for(int i = 0; i < vec1.size(); i++) ret += vec1[i] * vec2[i];
        return ret;
    }

    std::vector<double> col(const std::vector<std::vector<double>>& matrix, const int& col){       //OK
        if(matrix.empty() || col >= matrix[0].size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In col)" << '\n';
            #endif
            return std::vector<double> {};
        }
        std::vector<double> ret;
        for(int i = 0; i < matrix.size(); i++)
            ret.push_back(matrix[i][col]);
        return ret;
    }

    std::vector<std::vector<double>> multiply_of(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB){       //OK
        if(matrixA.size() == 1 && matrixA[0].size() == 1){
            auto ret = matrixB;
            for(int i = 0; i < matrixB.size(); i++)
                for(int j = 0; j < matrixB[0].size(); j++)
                    ret[i][j] *= matrixA[0][0];
            correct_false_0(ret);
            return ret;
        }

        if(matrixB.size() == 1 && matrixB[0].size() == 1){
            auto ret = matrixA;
            for(int i = 0; i < matrixA.size(); i++)
                for(int j = 0; j < matrixA[0].size(); j++)
                    ret[i][j] *= matrixB[0][0];
            correct_false_0(ret);
            return ret;
        }

        if(matrixA.empty() || matrixB.empty() || matrixA[0].size() != matrixB.size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In multiply)" << '\n';
            #endif
            return std::vector<std::vector<double>> {}; 
        }

        std::vector<std::vector<double>> ret;
        for(int i = 0; i < matrixA.size(); i++){
            ret.push_back(std::vector<double> {});
            for(int j = 0; j < matrixB[0].size(); j++)
                ret[i].push_back(dot(matrixA[i], col(matrixB, j)));
        }
        correct_false_0(ret);
        return ret;
    }

    std::vector<std::vector<double>> plus_of(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB){       //OK
        if(matrixA.empty() || matrixB.empty() || matrixA.size() != matrixB.size() || matrixA[0].size() != matrixB[0].size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In plus)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }
        std::vector<std::vector<double>> ret = matrixA;
        for(int i = 0; i < matrixA.size(); i++){
            for(int j = 0; j< matrixA[0].size(); j++){
                ret[i][j] += matrixB[i][j];
            }
        }
        correct_false_0(ret);
        return ret;
    }

    std::vector<std::vector<double>> minus_of(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB){       //OK
        if(matrixA.empty() || matrixB.empty() || matrixA.size() != matrixB.size() || matrixA[0].size() != matrixB[0].size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In minus)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }
        std::vector<std::vector<double>> ret = matrixA;
        for(int i = 0; i < matrixA.size(); i++){
            for(int j = 0; j< matrixA[0].size(); j++){
                ret[i][j] -= matrixB[i][j];
            }
        }
        correct_false_0(ret);
        return ret;
    }

    std::vector<std::vector<double>> power_of_n(const std::vector<std::vector<double>>& matrix, const int& pow){       //OK
        if(matrix.empty() || matrix.size() != matrix[0].size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In power_of_n)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }
        std::vector<std::vector<double>> ret = matrix;
        for(int i = 1; i < pow; i++)
            ret = multiply_of(ret, matrix);
        correct_false_0(ret);
        return ret;
    }

    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix){       //OK
        if(matrix.empty()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In transpose)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }
        std::vector<std::vector<double>> ret;
        for(int i = 0; i < matrix[0].size(); i++)
            ret.push_back(col(matrix, i));
        correct_false_0(ret);
        return ret;
    }

    std::vector<std::vector<double>> get_Identity_Matrix(const int& n){       //OK
        if(n <= 0){
            #ifdef DEBUG
            std::cout << "Invalid Input.(In get_Identity_Matrix)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }
        std::vector<std::vector<double>> ret;
        for(int i = 0; i < n; i++){
            ret.push_back({});
            for(int j = 0; j < n; j++){
                if(i == j) ret[i].push_back(1);
                else ret[i].push_back(0);
            }
        }
        return ret;
    }

    std::vector<std::vector<double>> append_Matrix_horizontally(const std::vector<std::vector<double>>& matrixA, const std::vector<std::vector<double>>& matrixB){       //OK
    //return [A,B]
        if(matrixA.empty() || matrixB.empty() || matrixA.size() != matrixB.size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In append_Matrix_horizontally)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }
        auto ret = matrixA;
        int n = matrixA.size();
        int m = matrixB[0].size();
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++)
            ret[i].push_back(matrixB[i][j]);
        }
        return ret;
    }


    void elem_row_1(std::vector<std::vector<double>>& matrix, int i, int j){       //OK
    //swap(row(i), row(j))
        if(matrix.empty() || i>=matrix.size() || j>=matrix.size() || i==j){
            #ifdef DEBUG
            std::cout << "Invalid Input.(In elem_row_1)" << '\n';
            #endif
            return;
        }
        swap(matrix[i], matrix[j]);
    }

    void elem_row_2(std::vector<std::vector<double>>& matrix, int i, double k){       //OK
    //row(i) *= k
        if(matrix.empty() || i>=matrix.size()){
            #ifdef DEBUG
            std::cout << "Invalid Input.(In elem_row_2)" << '\n';
            #endif
            return;
        }  
        for(int j = 0; j < matrix[0].size(); j++)
            matrix[i][j] *= k;
    }

    void elem_row_3(std::vector<std::vector<double>>& matrix, int i, int j, double k){       //OK
    //row(j) += k*row(i)
        if(matrix.empty() || i>=matrix.size() || j>=matrix.size() || i==j){
            #ifdef DEBUG
            std::cout << "Invalid Input.(In elem_row_3)" << '\n';
            #endif
            return;
        }
        for(int p = 0; p < matrix[0].size(); p++)
            matrix[j][p] += k* matrix[i][p];
        correct_false_0(matrix);
    }


    std::vector<std::vector<double>> get_steps_to_simplest_stair_matrix_by_row(std::vector<std::vector<double>> matrix){   //ok
    /*
    eg.
        ret={{1,1,2},{2,2,3},{3,1,2,0.5},......}
        {1,1,2}-> employ elem_row_1, i = 1, j = 2
        {2,2,3}-> ......
        ......

    */
    //定义一个返回值，用于存储每一步的操作和参数
        std::vector<std::vector<double>> ret;

        int row = 0;
        for(int col = 0; col < matrix[0].size(); col++){
            if(row >= matrix.size()) break;
            bool found = false;
            for(int cur_row = row; cur_row < matrix.size(); cur_row++){
                if(matrix[cur_row][col] != 0){
                    if(cur_row != row){
                        ret.push_back({1.0, (double)row, (double)cur_row});
                        elem_row_1(matrix, row, cur_row);
                    }
                    found = true;
                    break;
                }
            }
            if(!found) continue;


            if(matrix[row][col] != 1.0){
                double k = (1.0 / matrix[row][col]);
                ret.push_back({2.0, (double)row, k});
                elem_row_2(matrix, row, k);
            }

            for(int cur_row = 0; cur_row < matrix.size(); cur_row++){
                if(matrix[cur_row][col] != 0 && cur_row != row){
                    correct_false_0(matrix);
                    ret.push_back({3.0, (double)row, (double)cur_row, -matrix[cur_row][col]});
                    elem_row_3(matrix, row, cur_row, -matrix[cur_row][col]);
                }
            }
            row++;
        }
        return ret;
    }

    std::vector<std::vector<double>> transform_matrix_by_employing_given_steps_of_row_manipulation(const std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& steps){  //ok
        if(matrix.empty()){
            #ifdef DEBUG
            std::cout << "Invalid Input.(In transform_matrix_by_employing_given_steps_of_row_manipulation)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }

        auto ret = matrix;
        for(auto step: steps){
                if(step[0] == 1.0) elem_row_1(ret, step[1], step[2]); 
                if(step[0] == 2.0) elem_row_2(ret, step[1], step[2]);
                if(step[0] == 3.0) elem_row_3(ret, step[1], step[2], step[3]);
        }
        return ret;
    }

    std::vector<std::vector<double>> get_simplest_stair_matrix(const std::vector<std::vector<double>>& matrix){
        return transform_matrix_by_employing_given_steps_of_row_manipulation(matrix, get_steps_to_simplest_stair_matrix_by_row(matrix));
    }


    int rank_of(std::vector<std::vector<double>> matrix){   //ok   
        if(matrix.empty()){
            #ifdef DEBUG
            std::cout << "Invalid Input.(In rank)" << '\n';
            #endif
            return 0;
        }
        matrix = get_simplest_stair_matrix(matrix);
        int r = 0;

        for(int i = 0; i < matrix.size(); i++){
            for(int j = 0; j < matrix[0].size(); j++){
                if(matrix[i][j] > 1 - epsilon && matrix[i][j] < 1 + epsilon){
                    r++;
                    break;
                }
            }
        }
        return r;
    }

    std::vector<std::vector<double>> get_algebraic_complement(const std::vector<std::vector<double>>& matrix, int row, int col){       //OK
        std::vector<std::vector<double>> ret(matrix.size() - 1, std::vector<double>(matrix.size() - 1)); //！！！

        for (int i = 0, newRow = 0; i < matrix.size(); i++) {
            if (i == row) continue;

            for (int j = 0, newCol = 0; j < matrix[0].size(); j++) {
                if (j == col) continue;
                ret[newRow][newCol] = matrix[i > row ? (i - 1) : i][j > col ? (j - 1) : j];
                newCol++;
            }
            newRow++;
        }
        return ret;
    }

    double det(const std::vector<std::vector<double>>& matrix){       //OK
        if(matrix.empty() || matrix.size() != matrix[0].size()){
            #ifdef DEBUG
            std::cout <<"Invalid Input.(In det)" << '\n';
            #endif
            return 0;
        }
        double ret = 0;
        if(matrix.size() == 2)
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

        for(int i = 0; i < matrix.size(); i++){ //按第零行展开
            if(i % 2 ==0) ret-= matrix[0][i] * det(get_algebraic_complement(matrix, 0, i));
            else ret+= matrix[0][i] * det(get_algebraic_complement(matrix, 0, i));
        }
        return ret;
    }
    std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>>& matrix){  //ok
        if(matrix.empty() || matrix.size() != matrix[0].size()){
            #ifdef DEBUG
            std::cout << "Error: Invalid Input.(In inverse)" << '\n';
            #endif
            return std::vector<std::vector<double>> {};
        }

        auto steps = get_steps_to_simplest_stair_matrix_by_row(matrix);
        int rank = rank_of(matrix);
        if(rank == matrix.size()){
            return transform_matrix_by_employing_given_steps_of_row_manipulation(get_Identity_Matrix(matrix.size()), steps);        
            }
        else{
            std::cout << "Error: rank of matrix is " << rank <<", not matrix.size() = " << matrix.size() << ".(In inverse)" << '\n';
            return std::vector<std::vector<double>> {};
        }
    }

    void input_matrix(std::vector<std::vector<std::vector<double>>>& matrixs){
    /*
    实现一个输入矩阵的函数
    要求：各矩阵的大小自适应输入的元素行列数，输入方可以用空格来隔开矩阵间的元素
    按下enter时矩阵输入同步换行，连续按下两次enter表示结束当前矩阵的输入，连续按下三次enter代表完全终止输入，输入的矩阵依次存储于matrixs[0], matrixs[1],...中
    */
        int matrixCount = 0; // 记录已经输入的矩阵数量

        while (true) {
            std::cout << ">>" << (char)(matrixCount + 'A') << "=";

            std::vector<std::vector<double>> currentMatrix;
            std::string line;
            int consecutiveEmptyLines = 0; // 用于跟踪连续空行的次数
            int linecount = 0;

            while (consecutiveEmptyLines < 1) {
                if(linecount) std::cout << "    ";
                std::getline(std::cin, line);
                if (line.empty()) {
                    consecutiveEmptyLines++;
                } else {
                    consecutiveEmptyLines = 0; // 重置连续空行的次数
                    std::stringstream ss(line);
                    std::vector<double> row;
                    double num; 

                    while (ss >> num) {
                        row.push_back(num);
                    }

                    if (!row.empty()) {
                        currentMatrix.push_back(row);
                        linecount++;
                    }
                }
            }

            if (currentMatrix.empty()) {
                break; // 连续按下两次回车键表示结束输入
            }

            matrixs[matrixCount] = currentMatrix;
            matrixCount++;
        }
    }

    std::vector<std::vector<double>> parse_and_calculate(std::string s, std::vector<std::vector<std::vector<double>>>& matrixs){
        std::stack<std::vector<std::vector<double>>> operands;
        std::stack<char> operators;

        for(int i = 0; i < s.size(); i++)
            if(s[i] == '~') s[i] = '[';

        if(s[s.size() - 2] == '>' && (!(s[s.size() - 1] >= 'A' && s[s.size() - 1] <= 'Z') || s[s.size() - 3] != '-')){
            std::cout << "Invalid Input.(initial)" << '\n';
            return std::vector<std::vector<double>> {};
        }

        if(s[s.size() - 3] == '-' && s[s.size() - 2] == '>' && s[s.size() - 1] >= 'A' && s[s.size() - 1] <= 'Z'){
            auto result = parse_and_calculate(s.substr(0, s.size() - 3), matrixs);
            std::cout << "Result stored in " << s[s.size() - 1] << "." << '\n';
            matrixs[s[s.size() - 1] - 'A'] = result;
            matrixs[26] = result;
            return result;
        }

        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= 'A' && s[i] <= '[') {
                operands.push(matrixs[s[i] - 'A']);
            } else if (s[i] == '+' || s[i] == '*' || s[i] == '-') {
                if (i == 0 || s[i-1] == '(' || s[i-1] == '+' || s[i-1] == '*' || s[i-1] == '-') { // check if minus is unary
                    operators.push(s[i]);
                } else {
                    while (!operators.empty() && operators.top() != '(') {
                        char op = operators.top();
                        operators.pop();
                        if (op == '+') {
                            if (!operands.empty()) {
                                std::vector<std::vector<double>> matrixB = operands.top();
                                operands.pop();
                                if (!operands.empty()) {
                                    std::vector<std::vector<double>> matrixA = operands.top();
                                    operands.pop();
                                    operands.push(plus_of(matrixA, matrixB));
                                }
                            }
                        } else if (op == '*') {
                            if (!operands.empty()) {
                                std::vector<std::vector<double>> matrixB = operands.top();
                                operands.pop();
                                if (!operands.empty()) {
                                    std::vector<std::vector<double>> matrixA = operands.top();
                                    operands.pop();
                                    operands.push(multiply_of(matrixA, matrixB));
                                }
                            }
                        } else if (op == '-') {
                            if (!operands.empty()) {
                                std::vector<std::vector<double>> matrixB = operands.top();
                                operands.pop();
                                if (!operands.empty()) {
                                    std::vector<std::vector<double>> matrixA = operands.top();
                                    operands.pop();
                                    operands.push(minus_of(matrixA, matrixB));
                                }
                            }
                        }
                    }
                    operators.push(s[i]);
                }
            } else if (s[i] == '^') {
                int pow = s[i+1] - '0';
                i++;
                if (pow == 0) {
                    if (!operands.empty()) {
                        std::vector<std::vector<double>> matrix = operands.top();
                        operands.pop();
                        operands.push(inverse(matrix));
                    }
                } else {
                    if (!operands.empty()) {
                        std::vector<std::vector<double>> matrix = operands.top();
                        operands.pop();
                        operands.push(power_of_n(matrix, pow));
                    }
                }
            } else if (s[i] == 'd' || s[i] == 'r' || s[i] == 's' || s[i] == 't' || s[i] == 'i') {
                if (i + 1 < s.size() && s[i+1] == '(') {
                    int j = i + 2;
                    int cnt = 1;
                    while (j < s.size() && cnt > 0) {
                        if (s[j] == '(') cnt++;
                        else if (s[j] == ')') cnt--;
                        j++;
                    }
                    std::string sub_s = s.substr(i+2, j-i-3);
                    std::vector<std::vector<double>> matrix = parse_and_calculate(sub_s, matrixs);

                    double det_val = 0;
                    int rank_val = 0;
                    switch(s[i]){
                        case 'd':   det_val = det(matrix); 
                                    operands.push(std::vector<std::vector<double>>{{det_val}}); 
                                    break;
                        case 'r':   rank_val = rank_of(matrix);
                                    operands.push(std::vector<std::vector<double>>{{static_cast<double>(rank_val)}});
                                    break;
                        case 's':   operands.push(get_simplest_stair_matrix(matrix));
                                    break;
                        case 't':   operands.push(transpose(matrix));
                                    break;
                        case 'i':   operands.push(inverse(matrix));
                                    break;
                    }
                    i = j - 1;
                } else {
                    if (!operands.empty()) {
                        std::vector<std::vector<double>> matrix = operands.top();
                        operands.pop();

                        double det_val = 0;
                        int rank_val = 0;
                        switch(s[i]){
                            case 'd':   det_val = det(matrix); 
                                        operands.push(std::vector<std::vector<double>>{{det_val}}); 
                                        break;
                            case 'r':   rank_val = rank_of(matrix);
                                        operands.push(std::vector<std::vector<double>>{{static_cast<double>(rank_val)}});
                                        break;
                            case 's':   operands.push(get_simplest_stair_matrix(matrix));
                                        break;
                            case 't':   operands.push(transpose(matrix));
                                        break;
                            case 'i':   operands.push(inverse(matrix));
                                        break;
                        }
                    }
                }
            } 

            else if (s[i] == '(') {
                operators.push(s[i]);
            } else if (s[i] == ')') {
                while (!operators.empty() && operators.top() != '(') {
                    char op = operators.top();
                    operators.pop();
                    if (op == '+') {
                        if (!operands.empty()) {
                            std::vector<std::vector<double>> matrixB = operands.top();
                            operands.pop();
                            if (!operands.empty()) {
                                std::vector<std::vector<double>> matrixA = operands.top();
                                operands.pop();
                                operands.push(plus_of(matrixA, matrixB));
                            }
                        }
                    } else if (op == '*') {
                        if (!operands.empty()) {
                            std::vector<std::vector<double>> matrixB = operands.top();
                            operands.pop();
                            if (!operands.empty()) {
                                std::vector<std::vector<double>> matrixA = operands.top();
                                operands.pop();
                                operands.push(multiply_of(matrixA, matrixB));
                            }
                        }
                    } else if (op == '-') {
                        if (!operands.empty()) {
                            std::vector<std::vector<double>> matrixB = operands.top();
                            operands.pop();
                            if (!operands.empty()) {
                                std::vector<std::vector<double>> matrixA = operands.top();
                                operands.pop();
                                operands.push(minus_of(matrixA, matrixB));
                            }
                        }
                    }
                }
                if (!operators.empty() && operators.top() == '(') {
                    operators.pop();
                }
            }
        }

        while (!operators.empty()) {
            char op = operators.top();
            operators.pop();
            if (op == '+') {
                if (!operands.empty()) {
                    std::vector<std::vector<double>> matrixB = operands.top();
                    operands.pop();
                    if (!operands.empty()) {
                        std::vector<std::vector<double>> matrixA = operands.top();
                        operands.pop();
                        operands.push(plus_of(matrixA, matrixB));
                    }
                }
            } else if (op == '*') {
                if (!operands.empty()) {
                    std::vector<std::vector<double>> matrixB = operands.top();
                    operands.pop();
                    if (!operands.empty()) {
                        std::vector<std::vector<double>> matrixA = operands.top();
                        operands.pop();
                        operands.push(multiply_of(matrixA, matrixB));
                    }
                }
            } else if (op == '-') {
                if (!operands.empty()) {
                    std::vector<std::vector<double>> matrixB = operands.top();
                    operands.pop();
                    if (!operands.empty()) {
                        std::vector<std::vector<double>> matrixA = operands.top();
                        operands.pop();
                        operands.push(minus_of(matrixA, matrixB));
                    }
                }
            }
        }

        if (operands.empty()) {
            matrixs[26] = {};
            return std::vector<std::vector<double>> {};
        } else {
            matrixs[26] = operands.top();
            return operands.top();
        }
    }

    void single_matrix_input(std::vector<std::vector<double>>& dest, const std::string& first_row) {
        dest.clear();
        std::istringstream iss(first_row);
        std::vector<double> row;
        double num;
        while (iss >> num) {
            row.push_back(num);
        }
        if (!row.empty()) {
            dest.push_back(row);
        }
        while (1) {
            std::string line;
            std::cout << "    ";
            std::getline(std::cin, line);

            if (line.empty()) break;
            else {
                std::istringstream iss2(line);
                std::vector<double> row;
                double num;

                while (iss2 >> num) {
                    row.push_back(num);
                }

                if (!row.empty()) {
                    dest.push_back(row);
                }
            }
        }
    }

    void solve_equation_represented_as_an_enlarged_matrix(const std::vector<std::vector<double>>& matrix){
        //A为增广矩阵




    }

    void solve_equation_classical_format(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B){
        //solve Ax=B




    }

    void help(){
        std::cout <<    ">>This calculator has several useful functions.\n";
        std::cout <<    "  1.Input a single matrix: All letters from A-Z can be used to represent a matrix.\n";
        std::cout <<    "    For example, just type 'A=1 2', and press enter, then the calculator will recognize that you're inputting matrix A.\n";
        std::cout <<    "    In this case, '1 2' will be the first row of matrix A and after pressing enter, you can input the next row of matrix A.\n";
        std::cout <<    "    Note that you can input a matrix of any size you want.\n";
        std::cout <<    "    Just separate different elements by a white space. Press enter twice consecutively to end input.\n";
        std::cout <<    "  2.Input multiple matrices at the same time: Just type 'inputs' to initiate inputting matrix A-Z in order.\n";
        std::cout <<    "  3.Calculate any expression: Input any expression you want and the calculator will present the result.\n";
        std::cout <<    "    The calculator supports the following functions: + - * ^n i() r() d() s() t()\n";
        std::cout <<    "    i(): inverse, r(): rank, d(): determinant, s(): get row-reduced echelon, t(): transpose\n";
        std::cout <<    "    You may need to add more parentheses to ensure the result is correct.(Known bug)\n";
        std::cout <<    "    Here are some examples of valid inputs: d(A*B) A+B*C r(A*B*C)*C ...etc.\n";
        std::cout <<    "    Note: to calculate the product of a number and a matrix, just input the number into a matrix and use that matrix to represent the number.\n";
        std::cout <<    "  4. The 'ans': just use the symbol '~' to get the previous calculation result.\n";
        std::cout <<    "  5. The '->' operator: eg. input 'A+B->A', and the result of A+B will be stored in A.\n";

    }

    void parse_commands(const std::string& s, std::vector<std::vector<std::vector<double>>>& matrixs, int& command){
        //make sure to separate different functions by an empty line
        if(s == "help") command = 7;  //"help"
        
        else if(s.substr(0,5) == "solve"){ //"solve(...)"
            if(s[5] != '(') command = 1;
                else if(s.length() == 8) command = 5; //solve(A) mode
                     else command = 6; //solve(Ax=B) mode
        }

        else if(s.substr(0,6) == "inputs")  //"inputs"     initiate multiple matrixs input
            command = 3;

        else if(s[1] == '='){ // "A= 1 2 3 4"   initiate single matrix input
            if(!(s[0] >= 'A' && s[0] <= 'Z')) command = 1;
            else command = 2;
        }


        //This always stays at the end of parse_commands function, letting parse_and_calculate be the "default" option
        else command = 4;
    }


    void integrated_calculation(){
        std::cout << ">>Tip: type 'help' to see how to use this calculator.\n";
        std::string s;
        std::vector<std::vector<std::vector<double>>> matrixs(27);
        //command: 1: error, 2: input, 3: inputs, 4: move_on_to_calculate, 5: solve_equation_1, 6: solve_equation_2, 7: helps
        int command{};

        while(true){
            command = 0;
            std::cout << ">>";
            std::getline(std::cin, s);

            parse_commands(s, matrixs, command);
            switch(command){
                case 1: std::cout << "Invalid Input." << '\n';                                      break;
                case 2: single_matrix_input(matrixs[s[0] - 'A'], s.substr(2, s.length() - 2));      break;
                case 3: input_matrix(matrixs);                                                      break;
                case 4: print_matrix(parse_and_calculate(s, matrixs));                              break;
                case 5: solve_equation_represented_as_an_enlarged_matrix(matrixs[s[6] - 'A']);      break;
                case 6: solve_equation_classical_format(matrixs[s[6] - 'A'], matrixs[s[9] - 'A']);  break;
                case 7: help();                                                                     break;
            }

            if(command != 2) std::cout << "\n";
        }
    }

}

int main(){
    matcalc::integrated_calculation();
    return 0;
}

//TODO: add the function to solve linear equations.
//TODO: add the function to calculate the eigenvalues and eigenvectors of a matrix.
//TODO: add the function to calculate the singular value decomposition of a matrix.