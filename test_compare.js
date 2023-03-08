var {cosinesim, x_entropy, jensen_shannon, _kld} = require('./node_compare')
var {random} = require('mathjs')

function get_dist (len) {
    let dist = [];
    for (let i = 0; i < len; i++) {
        dist[i] = Math.random();
    }
    // let dist = random(Array(10), 0, 1)
    let sum = dist.reduce((partialSum, a) => partialSum + a, 0);
    // Normalize
    for (let i = 0; i < 10; i++) {
        dist[i] = dist[i]/sum
    }
    return dist
}

let len = 10
let dist1 = get_dist(len)
let dist2 = get_dist(len)

console.log(dist1.join(','))
console.log(dist2.join(','))
console.log('Cosine: ' + cosinesim(dist1, dist2))
console.log('KLD: ' + _kld(dist1, dist2))
console.log('JSD: ' + jensen_shannon(dist1, dist2))
console.log('XEn: ' + x_entropy(dist1, dist2))
