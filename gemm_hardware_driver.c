#define gemm_VERSION "1.0"
#define gemm_NAME    "gemm_drv"
#define INTERRUPT    54 //Linux Interrup Number (will need to change)
#define GEMM_BASE    0x12345678 //Base Address for hardware accelerator (will be changed)
#define GEMM_MASK    0x00ffffff // I don't exactly know what the mask is needed for yet
#define GEMM_SIZE    0x01000000 //Max Addresses (same as FPGA right now)
#define COMMAND_MASK 0x80000000 //needed for sending commands back and forth
#define 

MODULE_AUTHOR("ZaneLiam");
MODULE_LICENSE("GPL"); //??
MODULE_DESCRIPTION("GEMM Hardware Accelerator Driver");
#define DRIVER_NAME "gemm_hardware_accelerator"

static struct gemm_drv_local{
  int gemm_needed;
  unsigned long mem_start;
  unsigned long mem_end;
  volatile unsigned int *gemm_ptr;
  unsigned int offset;
//   struct proc_dir_entry *<gemm_file>; //Linux Directory for GEMM (probably not needed);
//   struct fasync_struct *fasync_fpga_queue ; idk what this does
} local_instance;


// DECLARE_WAIT_QUEUE_HEAD(fpga_wait); idk what this does

gemm_int_handler(); //needs to be implemented

//Opening the Gemm Hardware
static int gemm_open1 (struct inode *inode, struct file *file) {
   return 0;
}

//Releasing the Gemm Hardware
static int gemm_release1 (struct inode *inode, struct file *file) {
   return 0;
}

static int gemm_fasync1 (int fd, struct file *filp, int on)
{
//    printk(KERN_INFO "\ngemm_drv: Inside gemm_fasync \n"); Needed for linux! Commented out to prevent errors
   return fasync_helper(fd, filp, on, &local_instance.fasync_fpga_queue);

} 

//How to write to the Gemm Hardware
static ssize_t gemm_write1(struct file *filp, const char __user *buf, size_t count, loff_t *offp)
{
    int not_copied;
    printk(KERN_INFO "\nGEMM DRV MSG: Recieved WRITE CMD\n");  
    not_copied = copy_from_user((void *)l.fpga_ptr, buf, count);
    return count - not_copied;
}

//How to read from the Gemm Hardware
static ssize_t gemm_read1(struct file *filp, char __user *buf, size_t count, loff_t *offp)
{
    int not_copied;
    printk(KERN_INFO "\nGEMM DRV MSG: Recieved READ CMD\n");   
    not_copied  = copy_to_user(buf, (void *)l.fpga_ptr, count);
    return count - not_copied;
}


//This handles all IO operations from the User/Application
static long gemm_ioctl1(struct file *file, unsigned int cmd, unsigned long arg) {
   int retval = 0;
   unsigned long value;
   unsigned int command_type;
   unsigned int offset;
   volatile unsigned int *access_addr;
   printk(KERN_INFO "\nGEMM DRV MSG: Entering IO Control\n");

   offset = ~COMMAND_MASK & cmd & GEMM_MASK;//Find the offset 
   if(offset > GEMM_SIZE)
    //   retval=-EINVAL; idk what this does
   command_type = COMMAND_MASK & cmd;//Determines what the command is
   switch(command_type)
   {
      case 0:
         //read
         if(!access_ok(VERIFY_READ, (unsigned int *)arg, sizeof(int)))
            return -EFAULT;

	 value = readl((volatile unsigned int *)&l.fpga_ptr[offset]);
	 put_user(value, (unsigned long*)arg);
         printk("fpga_drv: Read value %08lx\n", value);
         break;

      case COMMAND_MASK:
         //write
         access_addr = l.fpga_ptr + offset;

         if(!access_ok(VERIFY_WRITE, (unsigned int *)arg, sizeof(int)))
            return -EFAULT;

         get_user(value, (unsigned long *)arg);
         writel(value, access_addr); 

#ifdef DEBUG
         printk("fpga_drv: Wrote value %08lx\n", value);
#endif
         break;

      default:
#ifdef DEBUG
         printk(KERN_ERR "fpga_drv: Invalid command \n");
#endif
         retval = -EINVAL;
   }

   return retval;
}
