// import {kldivergence} from 'mathjs'
// const { kldivergence  } = require('mathjs')
// const math = import('./math.js');

// this is how you do sums for some reason
function sum(x) {
    return x.reduce((partialSum, a) => partialSum + a, 0);
}

function cosinesim(x, y){
    var dotproduct=0;
    var mx=0;
    var my=0;
    for(let i = 0; i < x.length; i++){
        dotproduct += (x[i] * y[i]);
        mx += (x[i]*x[i]);
        my += (y[i]*y[i]);
    }
    mx = Math.sqrt(mx);
    my = Math.sqrt(my);
    return (dotproduct)/((mx)*(my));
}

// https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
function jensen_shannon (x, y) {
    if(x.length !== y.length) {
        throw Error('vector lengths do not match');
    }
    let m = [];
    for(let i = 0; i < x.length; i++){
        m[i] = (x[i] + y[i]) / 2;
    }
    return (math.kldivergence(x, m) + math.kldivergence(y, m)) / 2;
}

function x_entropy (x, y) {
    let s = []
    for(let i = 0; i < x.length; i++) {
        s[i] = x[i] * Math.log2(y[i])
    }
    return -1 * sum(s)
}

function default_scorer (x, y) {
    return (x === y) ? 1 : -1;
}

const directions = {"diagonal": 0, "left": 1, "up": 2}

// class grid_item {
//     constructor(dist, direction) {
//         this.edit_dist = dist
//         this.direction = direction
//     }
// }

// https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm#Constructing_the_grid
function needleman_wunsch_align(x, y, scorer = default_scorer) {
    // initialize the grid
    const grid = new Array(x.length);
    for(let i = 0; i < x.length + 1; i++) {
        grid[i] = [[-i, directions["up"]]];
        // grid[i][0] = [-i, directions["up"]];
    }
    for(let i = 1; i < y.length + 1; i++) {
        grid[0][i] = [-i, directions["left"]];
    }
    // console.log(grid)

    // fill in the grid
    for(let i = 1; i < x.length + 1; i++) {
        for(let j = 1; j < y.length + 1; j++) {
            // console.log(j);
            // console.log(y[j]);
            let z = grid[i-1][j-1][0] + scorer(x[i-1], y[j-1]);
            let item = [z, directions["diagonal"]];
            if(grid[i-1][j][0] - 1 > item[0]) { // up
                item = [grid[i-1][j][0] - 1, directions["up"]];
            }
            if(grid[i][j-1][0] - 1 > item[0]) { // left
                item = [grid[i][j-1][0] - 1, directions["left"]];
            }
            grid[i][j] = item;
        }
    }
    // console.log(grid);

    // traceback
    let tracer = [x.length - 1, y.length - 1];
    const alignment = [];
    alignment.push(tracer.slice());
    while(tracer[0] > 0 || tracer[1] > 0) {
        if(tracer[0] < 0 || tracer[1] < 0) {
            throw new Error("traceback out of bounds");
        }

        let direction = grid[tracer[0]][tracer[1]][1];
        if(direction === directions["diagonal"]) {
            tracer[0] = tracer[0] - 1;
            tracer[1] = tracer[1] - 1;
            // tracer.map((x) => x - 1)
            alignment.unshift(tracer.slice());
        }
        else if(direction === directions["up"]) {
            tracer[0] = tracer[0] - 1;
            alignment.unshift(tracer.slice());
        }
        else { // direction is left
            tracer[1] = tracer[1] - 1;
            alignment.unshift(tracer.slice());
        }
        // console.log(tracer);
    }
    // console.log(alignment);
    return alignment
}
