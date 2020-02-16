const request = require("request"),
    express = require("express"),
    router = express();
//template
// http://localhost:2718/search?query=long%20distance%20carpooling&mode=app&region=California,Virginia&age=all&price=5&year=2021

/**
 * Change this to post route later.
 */
router.get("/", (req, res)=>{
    var query = req.query.query,
        mode = req.query.mode,
        region = req.query.region,
        year = req.query.year,
        age = req.query.age,
        price = req.query.price;
    
    var search_data = {};

    var link = link = "https://www.google.com/search?q=" + query.toLowerCase();
    request(link, (err, body, response)=>{
        if(err || body.statusCode != 200)
        {
            throw new Error(err);
        }
        search_data.title_page = response.split("</title>")[0].split("<title>")[1];

        //Fuck this. I'll just use Python. 
        // to implement... Python crawler. JS wrapper.
        return res.send(response);
    });

});

module.exports = router;