const mongoose = require('mongoose');

const FileSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.ObjectId,
        ref: 'User',
        required: true
    },
    originalName: {
        type: String,
        required: true
    },
    filename: {
        type: String,
        required: true
    },
    size: {
        type: Number,
        required: true
    },
    mimetype: {
        type: String
    },
    status: {
        type: String,
        default: 'Processed'
    },
    uploadedAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('File', FileSchema);
