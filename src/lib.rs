use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Context {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue }
    }
}

#[derive(Debug)]
pub struct StorageBinding {
    pub data: Vec<u8>,
    pub buffer: wgpu::Buffer,
}

impl StorageBinding {
    pub fn new(context: &Context, data: Vec<u8>, output_buffer: bool) -> Self {
        let mut usage = wgpu::BufferUsages::STORAGE;

        if output_buffer {
            usage = usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        }

        Self {
            buffer: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data.as_slice(),
                    usage,
                }),
            data,
        }
    }

    pub fn new_empty(context: &Context, size: usize, output_buffer: bool) -> Self {
        let data = vec![0; size];

        let mut usage = wgpu::BufferUsages::STORAGE;

        if output_buffer {
            usage = usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        }

        Self {
            buffer: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data.as_slice(),
                    usage,
                }),
            data,
        }
    }
}

pub struct ComputeContext {
    pub compute_shader: wgpu::ShaderModule,
    pub storage_bindings: Vec<StorageBinding>,
    pub write_binding: usize,
}

impl ComputeContext {
    fn bind_group_layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        self.storage_bindings
            .iter()
            .enumerate()
            .map(|(i, _)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                count: None,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(1).unwrap()),
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                },
            })
            .collect()
    }

    fn bind_group_layout(&self, context: &Context) -> wgpu::BindGroupLayout {
        context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bind group layout"),
                entries: self.bind_group_layout_entries().as_slice(),
            })
    }

    fn pipeline_layout(
        &self,
        bind_group_layout: &wgpu::BindGroupLayout,
        context: &Context,
    ) -> wgpu::PipelineLayout {
        context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            })
    }

    fn compute_pipeline(
        &self,
        pipeline_layout: &wgpu::PipelineLayout,
        context: &Context,
    ) -> wgpu::ComputePipeline {
        context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline"),
                layout: Some(pipeline_layout),
                module: &self.compute_shader,
                entry_point: "main",
            })
    }

    fn get_write_binding_size(&self) -> u64 {
        self.storage_bindings
            .get(self.write_binding)
            .unwrap()
            .data
            .len() as u64
    }

    fn read_on_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.get_write_binding_size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn write_on_buffer(&self) -> &wgpu::Buffer {
        &self
            .storage_bindings
            .get(self.write_binding)
            .unwrap()
            .buffer
    }

    fn buffer_entries(&self) -> Vec<wgpu::BindGroupEntry> {
        self.storage_bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding.buffer.as_entire_binding(),
            })
            .collect()
    }

    fn bind_group(
        &self,
        bind_group_layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: bind_group_layout,
            entries: self.buffer_entries().as_slice(),
        })
    }
    fn encoder(&self, device: &wgpu::Device) -> wgpu::CommandEncoder {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    fn begin_compute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        compute_pipeline: &wgpu::ComputePipeline,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    fn read_data(&self, read_on_buffer: wgpu::Buffer, device: &wgpu::Device) -> Vec<u8> {
        let buffer_slice = read_on_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |x| {
            if x.is_err() {
                println!("{:?}", x)
            }
        });

        device.poll(wgpu::Maintain::Wait);

        buffer_slice
            .get_mapped_range()
            .chunks_exact(1)
            .map(|b| {
                let c: [u8; 1] = b.try_into().unwrap();
                c[0]
            })
            .collect::<Vec<u8>>()
    }

    pub async fn compute(&self, context: &Context) -> Vec<u8> {
        let device = &context.device;

        let bind_group_layout = self.bind_group_layout(context);

        let pipeline_layout = self.pipeline_layout(&bind_group_layout, context);
        let compute_pipeline = self.compute_pipeline(&pipeline_layout, context);

        let bind_group = self.bind_group(&bind_group_layout, device);

        let mut encoder = self.encoder(device);

        self.begin_compute_pass(&mut encoder, &bind_group, &compute_pipeline);

        let read_on_buffer = self.read_on_buffer(device);
        let write_on_buffer = self.write_on_buffer();

        encoder.copy_buffer_to_buffer(
            write_on_buffer,
            0,
            &read_on_buffer,
            0,
            self.get_write_binding_size(),
        );

        context.queue.submit(Some(encoder.finish()));

        self.read_data(read_on_buffer, device)
    }
}
